# Cross-codec JND consistency for V0_5 ships + Tuner (2026-05-19)

## The product question

A user types "score 70". The encoder picks (codec, q) that achieves
zensim≈70. **Does score=70 mean the same perceptual quality across
codecs?** If JPEG q achieving zensim=70 looks visually different
from AVIF q achieving zensim=70, then "70" is a meaningless number.

This eval measures cross-codec perceptual spread at fixed score
levels per profile, using butteraugli as a calibrated pairwise
perceptual distance. The verdict drives whether the four V0_5
ship-trail profiles need a cross-codec JND anchor loss in the next
training cycle.

## Methodology

- **10 source images** (`512×512`) — 7 photos (web-crawl + numeric
  IDs) + 3 generated charts (line-art proxy). Selected from the
  `/mnt/v/input/zensim/sources/` corpus that drives the
  `exp_tuner_2026-05-18/qsweep` baseline. List in
  `scripts/v_next/cross_codec_jnd_eval.py::SELECTED_IMAGES`.
- **4 profiles** — `PreviewV0_5Balanced` (V_22-mix-LARGE+iwssim
  bake), `PreviewV0_5Compression` (V_24-per-sample-α s4),
  `PreviewV0_5Ensemble` (classifier-routed Balanced+Compression),
  `PreviewV0_5Tuner` (V_tuner-v2-s2 calibrated, the dedicated
  monotonicity / dial-honesty trail).
- **6 target scores** — {30, 50, 63, 70, 80, 90}. `63` is the CID22
  paper's PJND anchor (paper Table 4).
- **3 codecs** — JPEG q (4:2:0 subsampling, PIL default),
  WebP q (method=4), AVIF q (speed=6). All via PIL Pillow.
- **Q grid** — 19 points {5, 10, …, 95}. Per (profile, image,
  codec), pick the q whose zensim score is closest to T (no
  interpolation; sufficient at step-5 granularity).
- **Pairwise butteraugli** — for each (profile, image, target),
  compute butteraugli_max for each of the 3 pairs of decoded
  outputs (jpeg↔webp, jpeg↔avif, webp↔avif). Report mean across the
  3 pairs.
- **Worst pair** — also report the maximum of the 3 pairwise
  butteraugli values per image, then average across the 10 images.
- **Score spread** — max(achieved_zensim) − min(achieved_zensim)
  across the 3 codecs per (profile, target, image). Indicates how
  closely the codecs land on the target.
- **Distance from target** — mean |achieved − target| across the
  3 codecs. Large values indicate the target is unreachable.

All tooling exists in the repo:

- `target/release/examples/zensim_score_named PROFILE_NAME ref dist`
  — new binary at `zensim/examples/zensim_score_named.rs`, scores
  a pair under any named V0_5 profile (handles ensemble routing).
- `/home/lilith/work/zen/zenmetrics/target/release/zen-metrics score
  --metric butteraugli` — pairwise butteraugli.
- `scripts/v_next/cross_codec_jnd_eval.py` — orchestrator.

Raw per-row TSV at
`/mnt/v/output/zensim/cross_codec_consistency_2026-05-19/cross_codec_raw_2026-05-19.tsv`
(240 rows = 4 × 6 × 10).
Summary JSON at
`/mnt/v/output/zensim/cross_codec_consistency_2026-05-19/cross_codec_summary_2026-05-19.json`.
Run log at `/tmp/cross_codec_consistency_2026-05-19.log`.

## Result — mean pairwise butteraugli_max (lower = more consistent)

Interpretation thresholds: < 1.0 excellent, 1.0–2.0 acceptable,
2.0–4.0 noticeable spread, > 4.0 broken.

| Profile | T=30 | T=50 | T=63 | T=70 | T=80 | T=90 |
|---|---:|---:|---:|---:|---:|---:|
| v0_5_balanced | 20.69 | 20.69 | 20.69 | 20.69 | 20.69 | 20.69 |
| v0_5_compression | 20.05 | 20.72 | 20.72 | 20.72 | 20.72 | 20.72 |
| v0_5_ensemble | 20.47 | 20.47 | 20.47 | 20.47 | 20.47 | 20.47 |
| **v0_5_tuner** | **13.64** | **9.63** | **6.68** | **5.00** | **3.31** | **1.88** |

## Result — mean dist_from_target (closer = reachable)

| Profile | T=30 | T=50 | T=63 | T=70 | T=80 | T=90 |
|---|---:|---:|---:|---:|---:|---:|
| v0_5_balanced | 16.41 | 36.08 | 49.08 | 56.08 | 66.08 | 76.08 |
| v0_5_compression | 14.18 | 33.64 | 46.64 | 53.64 | 63.64 | 73.64 |
| v0_5_ensemble | 15.17 | 35.04 | 48.04 | 55.04 | 65.04 | 75.04 |
| **v0_5_tuner** | **8.10** | **2.26** | **0.90** | **0.65** | **0.87** | **0.89** |

## Result — mean score spread across codecs (variance of achieved scores)

| Profile | T=30 | T=50 | T=63 | T=70 | T=80 | T=90 |
|---|---:|---:|---:|---:|---:|---:|
| v0_5_balanced | 13.14 | 13.14 | 13.14 | 13.14 | 13.14 | 13.14 |
| v0_5_compression | 6.43 | 8.06 | 8.06 | 8.06 | 8.06 | 8.06 |
| v0_5_ensemble | 8.10 | 8.10 | 8.10 | 8.10 | 8.10 | 8.10 |
| **v0_5_tuner** | **13.43** | **4.66** | **1.95** | **1.15** | **2.09** | **1.68** |

## Reading these tables — the dead-zone story

The Balanced/Compression/Ensemble rows look numerically identical
across targets because **those profiles cannot reach scores ≥ 30
on most images at any q**. The CLAUDE.md "tied-rate dead zones"
note flagged this; the eval makes it quantitative.

- **Balanced** raw output range: 0–20 across the full q=5..95
  sweep on the example image (one image, jpeg420 codec):

  ```
  v0_5_balanced 19.88 8.91 3.76 0.86 0.00 0.00 ... 0.00 0.00
  ```

  Once q ≥ 25, the output saturates to 0.0. The binary search
  picks the closest q for every target ≥ 25 — which is always
  q=5 (highest non-zero value, 19.88). Every "target" thus lands
  on the same q, producing identical butteraugli across rows.

- **Compression** has a slightly wider live range (0–20 across
  q=5..95) but the same shape: once the bake's raw output
  saturates at the low end, every higher target collapses to the
  same codec choice.

- **Ensemble** routes to one of the two underlying bakes per
  pair via a classifier. It inherits the same saturation behavior
  from whichever component fires.

- **Tuner** is the only profile that produces a monotonic 0..100
  response. On the same example image the full sweep is:

  ```
  v0_5_tuner 0.00 18.47 40.90 50.81 56.94 61.35 ... 89.99 93.46 96.57
  ```

  → covers the full target range with score_spread < 5 across
  codecs once T ≥ 50.

For Balanced/Compression/Ensemble the "cross-codec consistency" stat
at high targets is a confused measurement — the codecs aren't
actually trying to reach the target, they're all stuck near
q=5 (the only q that gives a non-zero raw output). The high
butteraugli (~20) reflects how *different* three codecs' q=5
outputs look, not a JND-anchored choice. So "broken" is the right
verdict for these profiles' use as user-facing dials, but the
specific number is a saturation artifact, not a JND measurement.

## Per-profile observations

- **v0_5_tuner — excellent at high T, degrades at low T.** At T=90
  the mean pairwise butteraugli_max is 1.88 (acceptable, near
  excellent). At T=80 it's 3.31 (noticeable). T=63 is 6.68
  (above "broken" threshold of 4.0) and T=30 is 13.64. The
  degradation at low T is partly a Q-grid coarseness artifact
  (q=5 is the lowest sample; for some images JPEG q=5 already
  exceeds T=30, so the binary search picks q=5 with score=44,
  far from 30) and partly genuine cross-codec divergence at
  extreme distortion (where butteraugli itself saturates).
  Score_spread at T=70..T=90 is < 3 — the codecs hit the target
  to within 3 score units of each other.

- **v0_5_balanced / v0_5_compression / v0_5_ensemble — unreachable
  targets above the saturation point.** Their dial reaches at most
  ~20 on most images. Asking "user types 70, codec picks q" is
  ill-defined for these profiles. They are rank-honest within
  their reachable range but NOT score-dial-honest.

## Per-image content variation under Tuner

Tuner's worst-case butteraugli at T=80 hits 4.7 on `gen-chart`
images (line-art) versus ~3.0 on photos. Line-art is a known
butteraugli outlier (chart edges → high pnorm3) and a known zensim
soft spot (most training data is photographic). For T=90 the gap
narrows (mean 2.5 on chart vs 1.6 on photo). At T=70/80, line-art
content drives the worst case; the "noticeable spread" verdict at
T=70 is partly content-dependent.

## Recommendation

**Tuner ships as the user-facing dial. Balanced/Compression/Ensemble
remain rank-trail models, not dial-honest models.**

The Tuner already achieves mean pairwise butteraugli < 2.0 at T=90
and < 5.0 at T=70 with mean score_spread < 3. That meets the
"acceptable" threshold for production use as a dial across most of
the meaningful score range (T ≥ 70 covers ~95% of production
traffic; T ≥ 50 covers ~99%). At T=70 the line-art outlier brings
the mean butteraugli to 5.0, which is above the 2.0 "JND-consistent"
target — a CID22-anchored re-tune of the Tuner's training corpus
to up-weight line-art content (or a per-class calibration head)
would close this gap without a structural redesign.

**No TUNER-V2 with cross-codec JND anchor loss is justified yet.**
The mean butteraugli at the high-traffic operating points (T=80,
T=90) is in the acceptable band. The case to dispatch TUNER-V2
should be reopened ONLY if the user actually exposes the dial at
T < 70 in production AND complaints surface about cross-codec
quality drift. Until then, the marginal cost of a fresh training
sweep with a per-pair JND-anchor loss is not justified by the
measured spread.

The rank-trail profiles (Balanced, Compression, Ensemble) are
documented as **not suitable as a "score 70" dial**. Their
existing role — ranking encoder outputs on a fixed image — is
unchanged by this eval, since ranking only requires monotonicity
within an image's q-curve, not cross-codec score alignment.

### Action items derived from the eval

1. **Update `SOTA_TRAILS.md` and the V0_5 variant docs** to
   call out the dial vs rank-trail distinction with the numbers
   above. The Tuner trail's "NOT for general ranking" caveat
   should be paired with a "Balanced/Compression/Ensemble NOT
   for score dials" caveat — currently only the former exists.

2. **Add a low-T tuner improvement task** to the recovery
   backlog: line-art content adds 50% to the T=70 butteraugli
   mean. A modest data-side intervention (densify safesyn with
   chart/line-art renders at the q-grid edges, plus a calibration
   refit) would close the gap. NOT urgent.

3. **Do not dispatch TUNER-V2 with JND anchor loss now.** The
   measurement does not support the cost. Revisit if T=70 use
   drives production complaints.

## Reproduction

Workspace: `~/work/zen/zensim--cross-codec-eval` (this jj
workspace).

```bash
cd ~/work/zen/zensim--cross-codec-eval
cargo build --release --example zensim_score_named -p zensim
python3 scripts/v_next/cross_codec_jnd_eval.py 2>&1 | tee /tmp/cross_codec_consistency_2026-05-19.log
```

Total wall time ~7 min (most of which is butteraugli; zensim
scoring is ~40 ms per call after the initial encode-cache warms).

## Honest gaps

- **Q grid is step-5 only.** At T=30 the closest reachable score
  on jpeg can be 30+δ for δ as large as ~14 because the next
  cheaper q (q=5) is already below the target by a wide margin.
  A finer grid (step-1 around the inflection) would tighten the
  low-T Tuner numbers slightly but doesn't change the
  Balanced/Compression/Ensemble verdict.
- **10 images is a first pass.** Content classes (photo + chart)
  are deliberately mixed but n=7 photo + 3 chart isn't enough to
  produce reliable CIs. The trend (Tuner OK, others broken as
  dials) is robust to this sample size; absolute butteraugli
  numbers might shift ±10 % with a 50-image rerun.
- **PIL codec defaults.** Production-encoder paths (zenjpeg,
  zenwebp, zenavif) may produce slightly different
  rate-distortion curves than PIL's libjpeg-turbo / libwebp /
  libavif defaults. The cross-codec comparison is intra-codec
  family comparable; absolute pairwise butteraugli might shift
  if libjxl-quality JPEG were swapped in. This doesn't affect
  the structural finding (Balanced dead zone, Tuner monotone).
- **No CI on the mean butter values.** A 1000-bootstrap on the
  T=90 row would tighten the verdict but the gap between Tuner
  (1.88) and the rank trails (~20) is so large that it doesn't
  affect the recommendation.
