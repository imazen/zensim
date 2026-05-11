# CID22 paper — page-by-page methodology checklist

**Source**: `/mnt/v/zen/zensim-training/2026-05-07/papers/CID22_wg1m99012.pdf`
(Sneyers / Ben Baruch / Vaxman, JPEG WG1 `wg1m99012`, 30 pages, 2023).

**Goal**: per Goal 2 of `PARITY_AND_METHODOLOGY_PLAN_2026-05-11.md`,
ensure every load-bearing methodology element in the paper is either
(a) enforced by an owning artifact in our pipeline, or (b) on the
explicit follow-up list. **Each row links to the page, the
methodology element, and the owning artifact.** Rows are written as
the paper is read in order.

Legend:
- ✅ enforced today
- ⏳ partially enforced (gap noted)
- ❌ NOT enforced (concrete follow-up listed)
- — informational, not load-bearing for our pipeline

---

## Page 1 — Title / abstract / intro

| Element | Status | Notes / owning artifact |
|---|---|---|
| 22,153 distorted images / 250 pristine 512×512 refs | — | informational; counts |
| 6 codec classes: JPEG, JPEG 2000, JPEG XL, HEIC, WebP, AVIF | — | informational; codec list |
| 1.4 M human opinions | — | informational |
| Quality range: **medium → near-visually-lossless** | ✅ | matches `CLAUDE.md` per-band rule (B0 < 50, B1 50–65, B2 65–90, B3 ≥ 90) |
| AIC-3 contribution (high to near-visually-lossless slot) | ✅ | our zensim shipping bar is "match-or-exceed ssim2", same AIC-3 regime |
| SSIMULACRA 2 introduced here | ✅ | we run fast-ssim2; Goal 3 reproduces these numbers |

## Page 2 — Related work

| Element | Status | Notes |
|---|---|---|
| AIC-1 / AIC-2 / AIC-3 scope diagram (Fig 1): LOW→HIGH / HIGH→VL / NVL→VL | — | clarifies that **CID22 sits in HIGH→VL**, NOT low quality |
| LIVE-IQA, TID2013, KADID-10k, PieAPP, KonJND-1k overview | — | inform our holdout strategy (we use TID, KADID, KonJND already) |
| **KADID10k is 95 % non-compression** | ✅ | flagged in `CLAUDE.md`'s training-goals section as integrity-only |
| **KonJND-1k is JPEG+BPG only, single-q-point** | ✅ | already in our pipeline as PJND anchor (≈63 ± 5) |
| DSIS (side-by-side) **not discriminative in HF range** | — | explains why TSBPC was preferred |
| Boosted triplet comparison [12] is the foundation of TSBPC | — | informational |

## Page 3 — Pairwise vs MOS protocols

| Element | Status | Notes |
|---|---|---|
| Pairwise → RMOS (Elo, 0..1), **separate per ref image** | ⏳ | our trainer uses RankNet *within-source*, the same property — but our trainer scales the loss against an absolute scalar (ssim2). Reasonable since ssim2 IS calibrated. |
| Thurstonian analysis maps RMOS → JND (just objectionable diff) | — | informational |
| DSIS-style absolute MOS scales across refs but needs many opinions | — | informational |
| Hybrid MOS + PC test [14] is the foundation | — | informational — we don't have humans, so no hybrid for us |
| **CID22 used a single batch (no active sampling)** | — | informational; their crowdsourcing simplification |

## Page 4 — Assessment protocols (load-bearing for reproduction)

| Element | Status | Notes |
|---|---|---|
| **TSBPC layout**: (R, A, B) shown; R on left, A↔B toggle on right; ternary "A best / B best / I can't choose" | — | not our protocol; we use metric ground-truth |
| TSBPC images **upscaled to fill screen**, ≥ 2 switches before submit | — | informational |
| **DSBQS layout**: single image, toggle to reference at `dpr1` resolution (1 px = 1 CSS px) | ✅ | informs our **viewing-condition feature** TODO (zensim issue #25 — crowdsourced eval app) |
| DSBQS scale 1/3/5/7/9 labels: very low / low / medium / high / very high | ✅ | matches our band cutoffs (B0–B3) in `CLAUDE.md` |
| **DSBQS 5 ≡ DSIS 4** ("perceptible but not annoying"; "medium quality; no annoying artifacts") | ✅ | confirms B1 (50–65) ↔ "medium quality with visible artifacts" |
| Slider starts at 5, increments of 0.5 (mouse) or 1 (keyboard) | — | informational |
| Anchors for DSBQS — **mozjpeg q30/q50/q70** + the ref itself + 10 distorted | — | informational; affects MOS calibration |

## Page 5 — Experiment setup (codec list — important for reproduction)

| Element | Status | Notes |
|---|---|---|
| **All refs are 512×512** (downsampled from Pexels stock) | ✅ | confirmed our 49 held-out refs are 512×512 (per stage-1 audit) |
| **15 content categories** | ❌ | we cluster by `zenanalyze` features (7 classes today). **Goal 4 subtask**: expand to 15 to match paper. Owner: balanced-holdout corpus, when built. |
| Codecs + versions (paper used these specific encoders): mozjpeg 4.1.0, Kakadu 8.2.2, libjxl 0.6.1, libheif/x265 2.8.0, libwebp 1.0.3, libaom 3.1.2, wzav1 1.0.2 | ⏳ | our pipeline uses NEWER versions (zenjpeg 0.5.4, libwebp 1.4.0, libjxl 0.10+, rav1e/zenavif latest). **Per-codec SROCC reproduction in Goal 3 must caveat the version delta**. |
| **8–11 quality settings per codec, fixed (not bitrate-targeted)** | ✅ | matches our pipeline — we sweep fixed q, not target bpp |
| Quality grid example: mozjpeg `cjpeg -quality {30,40,50,60,65,70,75,80,85,90,95}` | — | informational; we sample more densely (q step 5 from 0–70, step 2 from 70–100 per CLAUDE.md long-term) |
| 105,155 TSBPC triplets sampled randomly from "non-trivial" pairs, 10 opinions/triplet | — | informational; their pair-filtering rule |
| **Trivial pair filter** (paper p. 5): "0.5 bpp JPEG vs AVIF at > 1.5 bpp" considered trivial | ❌ | We don't yet filter trivial pairs from training. Most of our ranknet training is within-source which already prunes most trivial pairs, but cross-source trivials exist. **Follow-up**: add a "trivial-pair filter" gate in the trainer that drops pairs where butter and ssim2 disagree by < 1 unit (low-info pair). |
| DSBQS: **10 anchor distorted + ref-itself per source**, mozjpeg q30/q50/q70 as anchors | — | informational; our pipeline doesn't have DSBQS-style anchors today |

---

## Page 6 — Figure 2 (15 content categories, the canonical list)

The paper labels its 15 content categories by name; we should use
these exact names if our balanced-holdout is to be paper-comparable
(Goal 4 expansion target).

| Category | Refs |
|---|--:|
| animals | 11 |
| art-abstract-decoration | 16 |
| building-monument | 26 |
| diagram-chart | 13 |
| food-drinks | 26 |
| illustration-logo-text | 12 |
| indoors-rooms | 25 |
| landscape-nature | 23 |
| materials-clothes | 8 |
| night-nightlife | 18 |
| people-fashion | 18 |
| portrait | 10 |
| sky-clouds | 9 |
| sports | 17 |
| urban-industrial-cars | 18 |
| **Total** | **250** |

| Element | Status | Notes / owning artifact |
|---|---|---|
| **15 canonical category names + counts** | ❌ | Goal 4 balanced-holdout sampler currently clusters on `zenanalyze` features (7 classes). **Replace** with these 15 paper-aligned names; cluster CID22 refs first by visual inspection to label, then propagate the label scheme to our larger source corpus via nearest-neighbour. |
| Distribution skewed: building-monument and food-drinks each get 26, diagram-chart only 13 | ✅ | informs that balanced sampling must use stratified proportions, not uniform — Goal 4 corpus must respect these ratios |

## Page 7 — Crowdsourcing scale + outlier detection

| Element | Status | Notes |
|---|---|---|
| **1,071,300 TSBPC opinions / 35,710 sessions; 334,920 DSBQS opinions / 11,164 sessions** | — | informational, scale context |
| Honeypot screening: 2 obvious questions per 30-question session | — | informational |
| **TSBPC outlier rule**: per-session mean agreement < 0.25 → discard. **5,257 sessions (14.7 %) discarded**. | — | doesn't apply (no human raters); but DOES inform our `ssim2 ↔ butter` concordance filter at the same threshold magnitude |
| DSBQS outlier rules: (a) ref image gets score < 5, (b) > 20 % responses exactly = 5 (slider stuck), (c) mobile device, (d) abs(normalized diff) > 1 mean OR > 1 std | — | informational; KonJND-1k anchors have similar gates |
| First 3 scores of each DSBQS session also dropped (training effect) | — | informational |

## Page 8 — Bias correction, RMOS Elo, **monotonicity constraint** (LOAD-BEARING)

| Element | Status | Notes |
|---|---|---|
| **Bias correction**: per-session adjustment to zero out normalized-diff mean; clamp to `[0,10]` | — | informational; we don't do per-session, but per-image bias is mitigated by our averaging |
| **MCOS** = 10 × mean of bias-corrected scores per anchor. Scale `[0, 100]`. | ✅ | confirms our 0..100 zensim scale is paper-compatible |
| **Reference MCOS range: 82.5–92.6, mean 88.3** | ✅ | matches the V0_5 affine calibration target (already in `affine_calibrate_znpr_v2.py`) — refs aren't perfect 100, they cluster around 88.3 |
| **RMOS via Elo tournament**: A>B=2 wins for A, "I can't choose"=1 win each, normalized to `[0,1]` | — | informational |
| **Forced monotonicity in RMOS**: add **200 dummy "higher bitrate is better" opinions** per same-codec adjacent-bitrate pair | ✅ | **directly motivates our `--tv-weight` regularizer.** We use TV-on-adjacent-q pairs (lo<hi quality) — analogous mechanism, different implementation. |
| Worst-vs-best Elo tie smoothing: 10 % tie across all pairs to handle infinities | — | informational |

## Page 9 — Anchor adjustments (Table 1)

| Element | Status | Notes |
|---|---|---|
| **Table 1**: MCOS adjustments during interpolation, per anchor type | — | informational |
| **28 cases where q90 JPEG > reference score** | — | rare but happens; informs that even "the original" can score below an alias |
| ~5 % of MCOS scores adjusted; mean abs change 0.72; max 2.59 | — | informational |
| **RMOS=0 → anchor in 97 % of cases (87 % q30 JPEG)** | — | informational; informs that q30 JPEG is a reliable "minimum quality" anchor |

## Page 10 — Final MCOS distribution + monotonicity impact (LOAD-BEARING)

| Element | Status | Notes |
|---|---|---|
| **91.7 % of CID22 has MCOS ≥ 50** | ✅ | matches our pipeline focus (medium-quality-and-up) |
| MCOS distribution: bulk in MCOS 60 (medium-high) and 88 (near-lossless) | ✅ | informs our B2/B3 band emphasis |
| **Bootstrap CI**: 200 resamples → 90% CI width 4.457 (σ=1.254) | ❌ | We don't yet emit bootstrap CIs on our metric/SROCC reports. **Follow-up**: add `--bootstrap N` to `dataset_metric_baseline` to emit 90% CI alongside point SROCC. |
| **Effect of removing monotonicity constraint**: KRCC drops 0.937 → 0.5559 (40 % drop); SRCC drops 0.997 → 0.7417 (26 % drop) | ✅ | **VALIDATES our TV regularizer is doing the right thing.** Without monotonicity supervision, ranking accuracy degrades by ~25-40 %. |
| Effect of skipping bias correction alone: minor (KRCC 0.9411 vs 0.9361) | — | informs that bias correction is nice-to-have, not essential |

---

## Continued reading: pages 11–30 (queued)

Next subtasks:
- p. 11–15: Tables 2-4 + IQA metric overview
- p. 16–20: **Table 3 (per-metric SROCC) — Goal 3 reproduction target**
- p. 21–25: SSIMULACRA 2 architecture description
- p. 26–30: limitations + conclusions + references
