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

## Page 11 — MCOS distribution (Fig 3) + disagreement mitigation effect

| Element | Status | Notes |
|---|---|---|
| **Fig 3**: MCOS distribution by encoder, stacked histogram | — | informational; visualizes the codec-comparison shape |
| Bulk of CID22 in **MCOS 30–90 range**, peak around MCOS 88 (near-lossless) | ✅ | matches our pipeline's B2/B3 emphasis |
| Effect of skipping DSBQS-disagreement mitigation: SRCC 0.8868 / PCC 0.9013 / MAE 4.47 | — | informational; mild effect |
| Purpose of mitigation: complement incomplete TSBPC + resolve TSBPC↔DSBQS disagreements (Fig 4) | — | informational |

## Page 12 — Fidelity vs appeal disagreement (Fig 4)

| Element | Status | Notes |
|---|---|---|
| **Fig 4 example**: JPEG XL q60 ↔ AVIF aurora cq37, both ≈ 0.5 bpp. TSBPC says AVIF better (Elo 1695 vs 1552), DSBQS says JPEG XL better (MCOS 69.0 vs 59.3) | ⏳ | **Important caveat for Goal 3 reproduction**: when our zensim disagrees with paper SROCC for a particular pair, we can't always tell if WE'RE wrong vs the paper's MOS being a fidelity-vs-appeal corner case. **Follow-up**: when reproducing Table 3, flag cases where our score disagrees and check the (image-pair, ΔMCOS, ΔTSBPC) tuple — small ΔTSBPC = ambiguous ground truth. |
| Fidelity = faithfulness to reference; Appeal = "I prefer this even with detail loss" | ✅ | aligned: zensim is a **fidelity** metric, not appeal. Confirmed via the V0_5 KonJND anchor check. |
| AIC-3 CTC dataset [16] only measures appeal (no reference shown) | — | informational; different goal from CID22 |
| Final CID22 MCOS = 67.1 for JPEG XL, 56.7 for AVIF — DSBQS dominates the merge | ✅ | confirms the merge tilts toward fidelity (good for our use case) |

## Page 13 — AIC-3 image 4 fidelity-vs-appeal example (Fig 5)

| Element | Status | Notes |
|---|---|---|
| **Fig 5**: VVC at 2 JND looks "better" than reference because VVC denoises artistic noise in source | ✅ | confirms: a metric that scores "denoise improvement" as quality loss (which zensim does) matches fidelity |
| ΔTSBPC definition: `(#A>B) − (#B>A)` per triplet; larger = clearer consensus | — | informational |
| Image scores "agree" with comparison if preferred image has higher score (no tie) | — | informational |

## Page 14 — Table 2 (TSBPC↔MCOS agreement) + sample-size analysis

**Table 2 (load-bearing)**: agreement % between TSBPC consensus and MCOS, by mitigation strategy and ΔTSBPC. Excerpt:

| ΔTSBPC | none | monoton only | both | avg ΔMCOS | pairs |
|--:|--:|--:|--:|--:|--:|
| 1 | 64.6% | 54.1% | 56.3% | 1.98 | 12997 |
| 3 | 86.9% | 79.6% | 75.3% | 5.76 | 11168 |
| 5 | 96.8% | 93.5% | 88.9% | 10.08 | 10169 |
| 7 | 99.4% | 99.4% | 96.0% | 14.83 | 8500 |
| 10 | 100% | 99.8% | 99.4% | 19.26 | 5884 |
| 12 | 100% | 100% | 100% | 19.38 | 119 |

| Element | Status | Notes |
|---|---|---|
| **Empirical relationship: avg ΔMCOS ≈ 2 × ΔTSBPC** | — | informational; calibration insight |
| MCOS gap ≥ 20 → unanimous TSBPC consensus | ✅ | matches our shipping bar: target SROCC tightness in B2/B3 where most pairs differ by < 20 MCOS units |
| **Mitigations REDUCE the raw agreement** but improve calibration: monoton-only agreement is 54% at ΔTSBPC=1 (down from 64% raw); both mitigations = 56% | ⏳ | nuanced — monotonicity helps cross-image calibration but hurts raw within-image rank agreement for tiny gaps. **Implication for V_NEXT**: TV regularizer might also reduce raw agreement on tiny gaps; that's the price of cross-image calibration. Our per-band SROCC measurement is the correct way to detect this trade-off. |
| **Sample size guidance** (Fig 7): 80 DSBQS / 5 TSBPC per pair = within 90% CI of full | — | not directly applicable to our pipeline |

## Page 15 — Fig 7 sample size table + protocol improvements

| Element | Status | Notes |
|---|---|---|
| Fig 7: full RMSE table of (DSBQS%, TSBPC%) sample size vs full MCOS | — | informational |
| **80 DSBQS / 5 TSBPC = enough for 90% CI** | — | reduces future experiment cost |
| Future TSBPC variants: include `R` in triplets, A\|B + R\|R toggle, single-image-at-a-time + 3-button | — | informational |
| **Viewing-conditions not modeled** in CID22 (sRGB only, no HDR, no per-DPI variants) | ⏳ | **Goal 4 follow-up**: include a viewing-conditions-aware test, OR explicitly caveat that zensim is sRGB-only matching the paper |
| Mobile viewers excluded from CID22 | — | informational |
| AIC-3 CTC [16]: pairwise without reference → measures appeal only | — | informational |

---

## Page 16 — Encoder results: median + 5th-percentile bpp-MCOS curves

| Element | Status | Notes |
|---|---|---|
| Fig 8: median MCOS vs bpp per encoder | — | informational |
| Fig 9: 5th-percentile MCOS vs bpp ("worst-case") | — | informational; matches our "low-q sweep density" requirement |
| **Encoder-consistency framework**: pick encoder setting `M_avg − mσ ≥ M_min` (m=1 → 1/6 below threshold; m=2 → 1/50; m=3 → 1/1000) | ⏳ | Goal 6 site should expose σ alongside median for each encoder setting. Currently we only report per-bake SROCC, not per-encoder-setting σ. **Follow-up**: extend `dataset_metric_baseline` to emit per-encoder-setting σ. |

## Page 17 — Per-category encoder behaviour (Fig 10) + **objective metrics list**

| Element | Status | Notes |
|---|---|---|
| Fig 10: visual consistency (σ vs MCOS, by encoder) | — | informational; JPEG XL most consistent, AVIF/WebP less so |
| Per-category encoder ordering varies (diagram-chart: AVIF wins; landscape-nature: AVIF ≤ MozJPEG) | ✅ | confirms why Goal 4 balanced holdout needs per-category measurement |
| **Objective metric set (paper-canonical list)**: PSNR (ImageMagick 6.9.11), VMAF + SSIM + MS-SSIM + PSNR-Y + PSNR-HVS + CIEDE2000 (vmaf v2.3.0), Butteraugli + SSIMULACRA 1/2 (libjxl 0.8), LPIPS v0.1.4, DSSIM v3.2.0, FSIM v0.3.5 | ⏳ | We have **fast-ssim2** (our own) but not libjxl 0.8's ssim2. Goal 3 reproduction must caveat any delta between our fast-ssim2 and libjxl 0.8's. |

## Page 18 — Fig 11: per-category bpp-MCOS curves (visual reference only)

— Fifteen 3×5 grid of plots, one per content category. Visual.

## Page 19 — Fig 12: per-image curves for portrait category (10 images)

The **10 portrait refs** are explicitly named — useful for cross-
referencing our holdout: `pexels-photo-1933873`,
`pexels-photo-2598024`, `pexels-photo-2811087`, `pexels-photo-2846602`,
`pexels-photo-3155588`, `pexels-photo-3568544`, `pexels-photo-3586798`,
`pexels-photo-6996399`, `pexels-photo-7114620`, `pexels-photo-796526`.

| Element | Status | Notes |
|---|---|---|
| The "portrait" 10 refs include `pexels-photo-1933873` — also one of the **8 unblocked refs** flagged by our overlap audit | ✅ | Audit-trail: `pexels-photo-1933873` is in our holdout and the generator now blocks it. |

## Page 20 — **Table 3 / Table 4 / Table 5 — Goal 3 reproduction targets**

### Table 3 — Metric correlation with CID22 MCOS (all 250 refs)

| Metric | KRCC | SRCC | PCC |
|---|--:|--:|--:|
| **SSIMULACRA 2** | **0.6934** | **0.882** | **0.8601** |
| Butteraugli 2-norm | -0.6575 | -0.8455 | -0.8089 |
| Butteraugli 3-norm | -0.6547 | -0.8387 | -0.7903 |
| DSSIM | -0.6428 | -0.8399 | -0.7813 |
| VMAF | 0.6176 | 0.8163 | 0.7799 |
| FSIM | 0.6089 | 0.8005 | 0.7676 |
| PSNR-HVS | 0.6076 | 0.8100 | 0.7559 |
| Butteraugli max-norm | -0.5843 | -0.7738 | -0.7074 |
| SSIM | 0.5628 | 0.7577 | 0.7005 |
| MS-SSIM | 0.5596 | 0.7551 | 0.7035 |
| LPIPS | -0.5417 | -0.7316 | -0.6932 |
| SSIMULACRA 1 | -0.5255 | -0.7175 | -0.6940 |
| PSNR-Y | 0.4452 | 0.6246 | 0.5901 |
| PSNR (ImageMagick) | 0.3472 | 0.5002 | 0.4817 |
| CIEDE2000 | 0.3154 | 0.4584 | 0.4096 |

**Caveat**: 201/250 refs were in SSIMULACRA 2's training set. On the
**49 held-out** refs, paper-reported SSIMULACRA 2 is:
- **KRCC 0.7033 / SRCC 0.88541 / PCC 0.87448 / MAE 4.97**

That held-out number is the **Goal 3 hard target** for our
`fast-ssim2`. Reproduction tolerance per the plan: ±0.002 SROCC.

### Table 4 — Metric scores at KonJND-1k PJND threshold (mean ± stdev)

| Metric | BPG images | JPEG images |
|---|--:|--:|
| PSNR-Y | 39.61 ± 2.98 | 36.70 ± 3.79 |
| PSNR-HVS | 40.31 ± 1.78 | 39.96 ± 1.79 |
| SSIM (×100) | 98.55 ± 0.76 | 98.54 ± 0.81 |
| MS-SSIM (×100) | 99.21 ± 0.40 | 99.22 ± 0.38 |
| VMAF | 90.05 ± 2.25 | 91.86 ± 1.90 |
| **SSIMULACRA 2** | **65.38 ± 5.10** | **63.10 ± 4.65** |
| DSSIM (×1000) | 3.357 ± 1.267 | 3.817 ± 1.297 |
| Butteraugli 3-norm | 1.528 ± 0.192 | 1.699 ± 0.229 |
| PSNR (ImageMagick) | 35.17 ± 2.69 | 32.70 ± 3.32 |

**Used by us**: KonJND PJND check confirms our V0_5 calibration target
of ssim2 ≈ 63 ± 5 at PJND (matches the 63.10 ± 4.65 JPEG number).

### Table 5 — Quality-scale alignment (CID22 MCOS as canonical)

| Dataset / metric | medium (50) | high (65) | vis. lossless (90) |
|---|--:|--:|--:|
| **CID22 (MCOS)** | **50** | **65** | **90** |
| TID2013 (MOS) | 4.5 | 5.5 | 6 |
| KADID10k (DMOS) | 3.7 | 4.3 | 4.5 |
| KonFiG-IQA (F-JND) | 1.5 | 0.7 | 0 |
| AIC-3 (JND) | 3 | 1.7 | 0 |
| KonJND-1k (PJND) | — | — | 1 |
| PSNR-HVS | 35 | 40 | 50 |
| MS-SSIM (×100) | 98 | 99.2 | 99.8 |
| VMAF | 83 | 91 | 96 |
| DSSIM (×1000) | 8 | 3.5 | 1 |
| Butteraugli 3-norm | 2.5 | 1.6 | 0.5 |
| **SSIMULACRA 2** | **50** | **65** | **90** |

| Element | Status | Notes |
|---|---|---|
| **SSIMULACRA 2 maps 1:1 to CID22 MCOS** | ✅ | this is why we keep `score_ssim2` as the training target; matches `CLAUDE.md` per-band B0/B1/B2/B3 cuts |
| Band boundaries: **50 / 65 / 90** for medium / high / visually-lossless | ✅ | our `CLAUDE.md` bands match exactly |
| TID2013 MOS scale 0–9, mid≈4.5; KADID DMOS scale 1–5, mid≈3.7 | ✅ | inform per-band cutoffs when computing per-band SROCC for those datasets |
| KonJND-1k PJND ≈ 63 ± 5 on SSIM2 (from Table 4 JPEG row) | ✅ | matches our V0_5 calibration anchor |

---

## Pages 21–25 — Per-dataset scatter plots (Figs 13–17)

Pages 21-25 are 5 nine-panel scatter-plot figures showing metric-vs-MOS
correlations across 5 datasets. Each panel reports KRCC/SRCC/PCC for
one objective metric. These are **direct Goal 3 reproduction targets**
— numbers per metric and per dataset are extracted here.

### Fig 13 — CID22 (matches Table 3)

| Metric | KRCC | SRCC | PCC |
|---|--:|--:|--:|
| PSNR-Y | 0.4452 | 0.6246 | 0.5901 |
| PSNR-HVS | 0.6076 | 0.81 | 0.7559 |
| SSIM | 0.5628 | 0.7577 | 0.7005 |
| MS-SSIM | 0.5596 | 0.7551 | 0.7035 |
| VMAF | 0.6176 | 0.8163 | 0.7799 |
| **SSIMULACRA 2** | **0.6934** | **0.882** | **0.8601** |
| DSSIM | 0.6428 | -0.8399 | -0.7813 |
| Butteraugli 3-norm | 0.6547 | -0.8387 | -0.7903 |
| PSNR (ImageMagick) | 0.3472 | 0.5002 | 0.4817 |

### Fig 14 — TID2013 (compression rows ~10 %)

| Metric | KRCC | SRCC | PCC |
|---|--:|--:|--:|
| PSNR-Y | 0.4699 | 0.6394 | 0.428 |
| PSNR-HVS | 0.5464 | 0.6938 | 0.6846 |
| SSIM | 0.5707 | 0.7552 | 0.764 |
| MS-SSIM | 0.6068 | 0.7868 | 0.7802 |
| VMAF | 0.5608 | 0.7439 | 0.7728 |
| **SSIMULACRA 2** | **0.6322** | **0.8194** | **0.8103** |
| DSSIM | 0.6984 | -0.871 | -0.8021 |
| Butteraugli 3-norm | 0.4935 | -0.6639 | -0.4878 |
| PSNR (ImageMagick) | 0.4958 | 0.6869 | 0.6601 |

### Fig 15 — KADID10k (compression rows ~5 %)

| Metric | KRCC | SRCC | PCC |
|---|--:|--:|--:|
| PSNR-Y | 0.4555 | 0.6319 | 0.5932 |
| PSNR-HVS | 0.4229 | 0.5927 | 0.5949 |
| SSIM | 0.5889 | 0.7806 | 0.6576 |
| MS-SSIM | 0.6466 | 0.8359 | 0.6836 |
| VMAF | 0.5343 | 0.7253 | 0.7185 |
| **SSIMULACRA 2** | **0.587** | **0.7851** | **0.7018** |
| DSSIM | 0.6679 | -0.8561 | -0.6544 |
| Butteraugli 3-norm | 0.3846 | -0.543 | -0.4424 |
| PSNR (ImageMagick) | 0.4876 | 0.6757 | 0.6214 |

### Fig 16 — KonFiG-IQA (F-JND scale, all values negative)

| Metric | KRCC | SRCC | PCC |
|---|--:|--:|--:|
| PSNR-Y | 0.5841 | 0.7589 | 0.6966 |
| PSNR-HVS | 0.7767 | 0.9237 | 0.8454 |
| SSIM | 0.6126 | 0.7787 | 0.705 |
| MS-SSIM | 0.6605 | 0.8291 | 0.6832 |
| VMAF | 0.384 | 0.4896 | 0.4633 |
| **SSIMULACRA 2** | **0.7783** | **0.9273** | **0.8708** |
| DSSIM | 0.7563 | -0.914 | -0.6727 |
| Butteraugli 3-norm | 0.7679 | -0.9231 | -0.7584 |
| PSNR (ImageMagick) | 0.6501 | 0.8241 | 0.7216 |

### Fig 17 — AIC-3 CTC (excluding image 4, appeal-not-fidelity)

| Metric | KRCC | SRCC | PCC |
|---|--:|--:|--:|
| PSNR-Y | 0.4796 | 0.6406 | 0.6544 |
| PSNR-HVS | 0.6603 | 0.832 | 0.8264 |
| SSIM | 0.3701 | 0.5013 | 0.3646 |
| MS-SSIM | 0.6733 | 0.8411 | 0.8062 |
| VMAF | 0.6772 | 0.8416 | 0.8165 |
| **SSIMULACRA 2** | **0.7487** | **0.9012** | **0.89** |
| DSSIM | 0.7025 | -0.866 | -0.8218 |
| Butteraugli 3-norm | 0.6339 | -0.8076 | -0.7951 |
| PSNR (ImageMagick) | 0.4133 | 0.5604 | 0.578 |

| Element | Status | Notes |
|---|---|---|
| SSIMULACRA 2 is the **best on every one of the 5 datasets** | ✅ | Goal 3 reproduction target: our fast-ssim2 must reproduce these per-metric numbers to within ±0.002 SROCC |
| SROCC magnitudes vary widely across datasets (0.78 KADID → 0.93 KonFiG) | ✅ | because each dataset is dominated by different distortions; informs per-dataset SROCC reporting requirement |
| KonFiG and AIC-3 CTC are F-JND/JND scales, not MCOS | ✅ | informs that V_NEXT calibration's affine fit is only valid for MCOS-scale datasets (CID22, TID, KADID); KonJND needs a separate calibration |

---

## Continued reading: pages 26–30 (queued)

Next subtasks:
- p. 26–30: pairwise SROCC (Table 6), SSIMULACRA 2 architecture, limitations, conclusions
