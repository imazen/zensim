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

## Continued reading: pages 6–30 (queued)

Next subtasks:
- p. 6–10: scoring + bias correction (load-bearing for MCOS reproduction)
- p. 11–15: Table 3 numbers — `Goal 3` reproduction target
- p. 16–20: Table 5 + Table 6 — quality-scale anchors + pairwise SROCC
- p. 21–25: SSIMULACRA 2 architecture (Goal 3 / future zensim arch parity)
- p. 26–30: limitations + conclusions + references

Each next page-block adds 5-10 rows to this checklist. The ❌ rows
become explicit follow-up issues; ⏳ rows get a tracking note in the
relevant code.
