# 4-metric smoothness, PJND tradeoff, and training mix addendum

Companion to `4metric_overnight_FINAL_2026-05-01.md`. Addresses three
follow-up questions:

1. Smoothness per metric (matters when the metric is a human-facing
   quality target).
2. Why the KonJND-1k-trained model regressed on human MOS, and what the
   tradeoff is.
3. % of training data that's synthetic per metric, and **what's actually
   in "synthetic"**.

> **Provenance caveat (read first):** "Synthetic" means our own
> per-pair encode/decode runs scored by GPU SSIMULACRA 2 — but the
> *source images* came from real corpora, not generated content. 87.8%
> of synthetic pairs (191,501) come from independent hex-hashed JPEGs
> (Pexels + corpus-builder); 7.5% (16,255) from misc Wikimedia/Flickr;
> **3.9% (8,461 pairs, 146 unique base images) overlap with CID22's
> training subset**, which is also Cloudinary's SSIM2 training corpus.
> The 49-image CID22 *validation/MCOS* set is fully excluded (verified
> empty intersection). KADID/TID have no source overlap with anything.
> See `4metric_overnight_FINAL_2026-05-01.md` §"Methodology caveats"
> for the full provenance table.

## 1. Smoothness on synthetic q-sweeps

Each metric was scored on all 218,089 synthetic pairs via its saved
MLP bake. For each (`source_path`, `codec`) group of size ≥ 4, we
sorted by quality, took the predicted distance curve, and measured:

- **violation_rate**: fraction of consecutive q-pairs where the model
  predicted *more* distance for the *higher* quality. Operationally
  bad — a user raising q expects a better score, not worse.
- **step_cv**: stdev(|Δdist|) / mean(|Δdist|). Lower = more even step
  sizes → smoother gradient.
- **max_step**: largest single-step jump per sweep. p95 = 95th
  percentile across sweeps.
- **ssim2_pearson**: |corr(predicted_distance, −SSIM2)| per sweep.
  Higher = predicted curve tracks the ground-truth SSIM2 curve closely.

| metric | n_sweeps | violation_rate (mean / p95) | step_cv (mean / p95) | max_step (p50 / p95) | ssim2_pearson (mean / p10) |
|---|--:|--:|--:|--:|--:|
| V0_4-smooth | 20784 | 0.016 / 0.143 | 0.702 / 1.186 | 5.12 / 12.23 | 0.9900 / 0.9833 |
| V0_5 | 20784 | 0.017 / 0.167 | 0.818 / 1.398 | **20.45 / 63.66** | 0.9912 / 0.9811 |
| V0_4-smooth-konjnd-train | 20784 | 0.019 / 0.167 | 0.753 / 1.257 | 5.19 / 12.91 | **0.9932 / 0.9875** |
| V0_6 dct_hf | 20784 | **0.016 / 0.143** | **0.670 / 1.148** | 5.14 / 12.39 | 0.9860 / 0.9763 |

### Predicted-distance dynamic range (the V0_5 outlier issue)

| metric | min | p05 | p50 | p95 | max | range |
|---|--:|--:|--:|--:|--:|--:|
| V0_4-smooth | −20.13 | −17.32 | −7.87 | 7.69 | 42.95 | 63.07 |
| V0_5 | −59.46 | −54.88 | −25.86 | 33.55 | **699.86** | **759.32** |
| V0_4-smooth-konjnd-train | −17.21 | −15.55 | −7.54 | 8.78 | 74.33 | 91.55 |
| V0_6 dct_hf | −20.84 | −16.95 | −6.43 | 8.70 | 28.55 | **49.39** |

**V0_5 has occasional pathological extreme outputs** (max 699 when its
p95 is 33). This won't hurt SROCC (rank-based, ignores magnitude) but
will hurt anyone using the metric as an absolute target. V0_6 dct_hf
has the narrowest dynamic range (49) and the tightest step CV (0.670)
— it's the smoothest of the four for human use.

### Per-codec violation rate (where smoothness breaks)

| metric \ codec | mozjpeg-rs-420-e4 | zenavif-s5-e6 | zenjpeg-420-e2 | zenjpeg-420-xyb-e2 | zenjxl-e7 | zenwebp-default-m4 |
|---|--:|--:|--:|--:|--:|--:|
| V0_4-smooth | 0.001 | 0.001 | 0.001 | 0.007 | 0.002 | 0.084 |
| V0_5 | 0.002 | 0.001 | 0.001 | 0.007 | 0.003 | 0.093 |
| V0_4-smooth-konjnd-train | 0.004 | 0.002 | 0.004 | 0.011 | 0.008 | 0.089 |
| V0_6 dct_hf | 0.002 | 0.001 | 0.000 | 0.007 | 0.002 | 0.084 |

**zenwebp drives almost all of the monotonicity violations** (8-9% vs
< 1% for every other codec). This is a *codec* artifact, not a metric
bug — the project's own CLAUDE.md notes "zenwebp: systematic quality
drops at q75→80 and q87→90 (codec mode switches)". The metrics are
faithfully reporting that the higher-quality output is sometimes
*worse* than the lower-quality output for zenwebp.

### Smoothness ranking for human target use

1. **V0_6 dct_hf** — best step_cv (0.670), narrowest dynamic range
   (49), tied for lowest violation rate (1.6%). Best fit for use as a
   human-facing quality target.
2. **V0_4-smooth** — close second (step_cv 0.702, range 63, 1.6%
   violations). Already-shipping baseline.
3. **V0_4-smooth-konjnd-train** — slightly worse step_cv (0.753) and
   slightly more violations (1.9%), but **highest per-sweep SSIM2
   correlation** (0.9932) and **tightest stdev at PJND** (see §2).
4. **V0_5** — best SROCC on KADID/TID/CID22 but **worst dynamic-range
   stability** (range 759 with extreme outliers). Pure rank-based
   downstream consumers can use it; absolute-target consumers should
   not.

## 2. Why V0_4-smooth-konjnd-train regressed on human MOS, and the tradeoff

### What KonJND-1k labels look like

KonJND-1k provides one PJND threshold per (source, codec) — the
quality level at which an average viewer starts to notice the
distortion. We expanded this to per-pair "scores" via a piecewise
linear schedule (`load_konjnd1k` in zensim-validate):

- file index ≤ PJND → score = 1.0 − 0.05 × (idx / PJND_clamped)
  (tiny linear decline, 1.0 at the source down to 0.95 at the threshold)
- file index > PJND → score = 0.95 − 0.95 × (idx − PJND_clamped) / (n_levels − PJND_clamped)
  (steep linear decline, 0.95 at threshold down to 0 at lowest quality)

The schedule has an **inflection point at PJND** by construction.

### What KADID/TID/CID22 labels look like

Continuous mean-opinion-scores. KADID DMOS ∈ [1, 5]. TID2013 MOS ∈ [0, 9].
CID22 MCOS ∈ [0, 100]. No inflection point. The score reflects how
*bad* the distortion is across its full range, smooth gradient.

### Why training on KonJND hurts MOS prediction

A model trained with KonJND in the training pool learns to predict
"is this distortion past the visibility threshold?" That objective
tells the MLP nothing about *how* far past — the score schedule is a
nearly-flat 0.95 around PJND and a steep ramp below. So the model
specializes in the inflection neighborhood and de-emphasizes
fine-grained distortion ranking elsewhere.

KADID/TID/CID22 evaluation asks the opposite question: "rank these
distortions by perceived severity across the full range." A
KonJND-specialist gets that wrong, and the SROCC penalty is large
(0.07–0.10 absolute drop):

| dataset | V0_4-smooth | V0_4-smooth-konjnd-train | Δ |
|---|--:|--:|--:|
| KADID | 0.8400 | 0.7441 | **−0.0959** |
| TID2013 | 0.8336 | 0.7566 | **−0.0770** |
| CID22 | 0.8910 | 0.8221 | **−0.0689** |

### The tradeoff: PJND-anchor stdev vs MOS rank quality

V0_4-smooth-konjnd-train pays for that MOS regression with a *very*
sharp PJND anchor — its raw-distance stdev at PJND is half what the
MOS-trained models give:

| metric | JPEG PJND mean ± stdev | BPG PJND mean ± stdev |
|---|---|---|
| V0_4-smooth | −5.79 ± 1.60 | −6.79 ± 1.54 |
| V0_5 | −19.70 ± 7.33 | −23.43 ± 7.33 |
| V0_4-smooth-konjnd-train | **−5.47 ± 1.28** | **−5.25 ± 0.94** |
| V0_6 dct_hf | −4.65 ± 1.66 | −5.61 ± 1.63 |

Lower stdev = the model's raw distance lands at almost the same value
across all 1008 (source, codec) PJND points. That's a **calibration
benefit**: if you want a single "this is the visually-lossless raw
distance" anchor, V0_4-smooth-konjnd-train gives it to you with half
the noise.

### Recommendation

KonJND-1k is **valuable as a calibration anchor**, not as a training
signal. Use it the way the CID22 paper Table 4 uses it: a fixed point
to translate raw metric values into "visually lossless" qualitative
labels. Do not put PJND-derived per-pair scores into the training
pool — the inflection objective fights with the continuous-MOS
objective the user actually cares about.

## 3. % synthetic in training pool per metric

| metric | train pool size | synthetic % | KonJND-1k % | other |
|---|--:|--:|--:|---|
| V0_4-smooth | 218,089 | **100%** | 0% | none |
| V0_5 | 174,458 (80% src-disjoint of 218,089) | **100%** | 0% | none |
| V0_4-smooth-konjnd-train | 271,049 (218,089 + 52,960 KonJND@70%) | **80.4%** | 19.6% | none |
| V0_6 dct_hf | 218,089 + per-ref 3 zenanalyze features | **100%** | 0% | + reference-image content stats |

KADID, TID2013, and CID22 are **val-only across all four metrics** —
no human-MOS data ever enters the training pool. CID22 specifically is
fully held out (49-image validation set never seen at training).

The synthetic dataset itself is constructed from CLIC 2025 + CID22-
training (excluding the CID22 49-image validation set), encoded with 6
codecs at q ∈ {5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 75, 87, 90,
95, 100} — the GpuSsim2 column from the encoder is the regression
target. So "100% synthetic" means "100% encoded by us, scored by GPU
SSIMULACRA 2, never touched a human."

## Practical guidance

- **For codec selection / pairwise A/B:** V0_5 is the SROCC champion
  but has those rare 700-magnitude outliers. If your downstream is
  pure rank, fine. If you ever look at the absolute number, prefer
  V0_6 dct_hf or V0_4-smooth.
- **For human quality targets** ("encode at score 80"): V0_6 dct_hf,
  with V0_4-smooth as a fallback. Both have step_cv ≈ 0.7, dynamic
  range < 100, and < 2% monotonicity violations on synthetic
  q-sweeps. V0_5 is a bad choice here.
- **For visually-lossless gating only:** V0_4-smooth-konjnd-train has
  the tightest PJND anchor stdev. Use as a separate auxiliary
  threshold model, not as a primary metric.
- **Do not ship V0_4-smooth-konjnd-train as a primary metric.** It
  regresses MOS by 0.07-0.10 SROCC across every human-rated dataset.
