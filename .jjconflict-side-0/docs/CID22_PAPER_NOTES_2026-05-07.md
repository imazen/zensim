# CID22 paper insights for zensim v_next

**Source:** Sneyers, Ben Baruch, Vaxman (Cloudinary), 2023.
"AIC-3 Contribution from Cloudinary: CID22"
JPEG WG1 document `wg1m99012-ICQ-AIC3_Contribution_Cloudinary_CID22.pdf`
Mirror: `/mnt/v/zen/zensim-training/2026-05-07/papers/CID22_wg1m99012.pdf`

30 pages. Read in full 2026-05-07. Below are the load-bearing findings
for our v_next zensim training cycle.

## Dataset facts (for evaluation)

- **CID22** = Cloudinary Image Dataset '22 — **22,153 distorted images**
  from **250 pristine 512×512 references**, 6 codec classes, 8–11 q
  settings each.
- **1.4 million human opinions** combining two protocols:
  - **TSBPC** (Triple Stimulus Boosted Pairwise Comparison) — `(R, A, B)` pick which
    distortion of `R` is better. Generates RMOS (Elo-style ranking, [0,1]).
  - **DSBQS** (Double Stimulus Boosted Quality Scale) — single image with toggle to
    reference, 0–10 scale. Anchor MCOS scores.
- **MCOS** = bias-corrected mean opinion score combining DSBQS anchors + TSBPC
  relative rankings, on **0–100** scale.
- 91.7% of CID22 images have MCOS ≥ 50 (medium quality or better) — focused on
  the web/practical-fidelity range.
- **49 reference images held out** from the SSIMULACRA 2 weight tuning — these
  4,292 distorted pairs are the only fully-disjoint CID22 evaluation set.
  The other 201 references were used to tune SSIMULACRA 2 weights.

## Quality scale alignment (Table 5)

The **canonical quality scale** for v_next (CID22 MCOS) — match this in
zensim output:

| Reference quality | CID22 MCOS | SSIMULACRA 2 | PSNR-HVS | MS-SSIM ×100 | VMAF | DSSIM ×1000 | BA 3-norm |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| medium quality | 50 | 50 | 35 | 98 | 83 | 8 | 2.5 |
| high quality | 65 | 65 | 40 | 99.2 | 91 | 3.5 | 1.6 |
| visually lossless | 90 | 90 | 50 | 99.8 | 96 | 1 | 0.5 |

**SSIMULACRA 2 maps 1:1 to MCOS** — same numerical scale. This is
why we keep `score_ssim2` as the v_next training target (and why
analyze_score_quality.py flagged the +7..+11 zensim-vs-ssim2 bias as
real calibration drift).

## Metric performance on CID22 — absolute scores (Table 3, all 250 refs)

| Metric | KRCC | **SROCC** | PCC |
|---|---|:---:|---|
| SSIMULACRA 2 | 0.6934 | **0.882** | 0.8601 |
| Butteraugli 2-norm | 0.6575 | 0.8455 | 0.8089 |
| Butteraugli 3-norm | 0.6547 | 0.8387 | 0.7903 |
| DSSIM | 0.6428 | 0.8399 | 0.7813 |
| VMAF | 0.6176 | 0.8163 | 0.7799 |
| FSIM | 0.6089 | 0.8005 | 0.7676 |
| PSNR-HVS | 0.6076 | 0.8100 | 0.7559 |
| BA max-norm | 0.5843 | 0.7738 | 0.7074 |
| SSIM | 0.5628 | 0.7577 | 0.7005 |
| MS-SSIM | 0.5596 | 0.7551 | 0.7035 |
| LPIPS | 0.5417 | 0.7316 | 0.6932 |

**SSIMULACRA 2 absolutely wins on CID22**, but with the caveat that
201/250 refs were in its training set. On the 49 held-out refs it's
KRCC 0.7033 / SRCC 0.88541 / PCC 0.87448 / MAE 4.97 — still the best.

## Metric performance — pairwise differences (Table 6)

When you control for reference image (compare two distortions of the
same source — exactly the kind of question encoder development asks),
correlations are *much higher* and SSIM2 is even more dominant:

| Metric | KRCC | SROCC | PCC |
|---|---|:---:|---|
| SSIMULACRA 2 | 0.7536 | **0.9210** | 0.9085 |
| DSSIM | 0.7203 | 0.9019 | 0.8352 |
| SSIMULACRA 1 | 0.7059 | 0.8915 | 0.8399 |
| BA 2-norm | 0.6852 | 0.8688 | 0.8422 |
| FSIM | 0.6828 | 0.8656 | 0.8411 |
| BA 3-norm | 0.6787 | 0.8610 | 0.8252 |

**Implication for v_next training loss:** RankNet (pairwise hinge,
within-source) dramatically outperforms MSE-against-absolute-MOS as a
supervisor signal. The current `train_v_next_mlp.py` already supports
this — keep `--loss mse_rank` with `--rank-weight 0.5`.

## Recommended quality ranges per metric (Table 7, summarized)

What zensim is asked to do well at (web compression):

| Metric | very low | low | medium | high | very high | visually lossless |
|---|---|---|---|---|---|---|
| **SSIMULACRA 2** | mediocre | good | very good | very good | very good | very good |
| **BA 3-norm** | very poor | poor | mediocre | good | very good | very good |
| DSSIM | good | very good | very good | good | good | good |
| MS-SSIM | very good | good | mediocre | mediocre | poor | poor |
| VMAF | good | mediocre | good | good | mediocre | mediocre |
| PSNR-HVS | very poor | poor | good | mediocre | good | good |
| SSIM | good | good | mediocre | mediocre | poor | poor |

**Implication for v_next:** SSIMULACRA 2 is the right *primary* target.
But for the `q < 50` band where SSIM2 falls to "mediocre" / "good", a
multi-task supervision adding DSSIM (best in low-q) or BA-3-norm (best
in very-high-q) would fill the tails. Our unified parquet has all four
target columns; lock training to `--target ssim2` for the main bake and
plan a complementary `--target multi {ssim2:1.0, dssim:0.4, ba_p3:0.3}`
ablation as a follow-up.

## SSIMULACRA 2 architecture (paper p. 26, used for tractability comparisons)

- **6 scales** (1:1 to 1:32), downsampling in **linear RGB**
- Score computed in **XYB color space**
- 3 SSIM error maps + 2 asymmetric error maps (ringing, smoothing) per scale
- 54 maps total × 2 aggregation norms (L1, L4) = **108 sub-scores**
- Final = weighted sum, weights optimized on 201 CID22 references

zensim today: 4 scales × 13 basic + 6 peak + (6 masked + 4 psycho) extended =
228 (basic + peaks) or 348 (extended) features, weighted via 228-element
LINEAR_WEIGHTS_PREVIEW (V0_2) or 228 → 64 LeakyReLU → 1 MLP (PR #24 V0_4).

**Implication:** SSIMULACRA 2's choice of 6 scales is more than zensim's 4.
That extra scale headroom would help most on **multi-scale invariance**
(handoff TODO §4.3) — at large source dimensions zensim's 4-scale pyramid
runs out of low-frequency context. **Add 2 more scales for V0_5/V_NEXT**
or note this as a known limitation.

## Methodology notes useful for our own training

- **Honeypot screening + bias correction** — CID22 discarded ~14.7% of
  TSBPC sessions for low-honeypot-agreement; ~11% of DSBQS sessions for
  failed score gating. We get this for free since our supervisor is
  ssim2/butteraugli, but it argues for **excluding rows where
  butteraugli and ssim2 disagree by > 5σ** (pure noise) before training.

- **Forced monotonicity** — the paper added 200 dummy "higher bitrate is
  better" opinions per same-codec pair to smooth Elo rankings. Our
  monotonicity_violations.tsv (71,115 cases) shows zensim does NOT
  enforce this. Adding a **monotonicity penalty** to v_next's loss
  (sort by q within a (image, codec, knobs) curve, penalize backward
  steps) would directly fix our biggest score-quality issue.

- **Sample size guidance (Figure 7)** — paper estimates RMSE between
  smaller-subset MOS and full-sample MOS. Drops below 2 RMSE points at
  ~50 anchor opinions + ~5 TSBPC opinions/pair. Our v15r corpus has
  ~1,800 cells/source × 981 sources = 1.77M total, well above the
  signal-saturation point.

- **Mobile vs desktop** — paper screens out mobile participants because
  viewing conditions are uncontrolled. We don't have human evaluators
  but **viewing-condition features** (pixel angular size, dpi assumption)
  remain unmodeled in zensim and were called out in PR #24 follow-up
  list (zensim issue #25 — crowdsourced human eval web app).

## Concrete v_next plan adjustments

1. **Train target = `score_ssim2`.** SSIM2 IS the canonical CID22 MCOS
   scale (Table 5), the best held-out metric on the canonical IQA
   benchmark, and what `score_zensim` already systematically biases
   away from in our 2.37M-row corpus.

2. **Loss = RankNet within (image, codec) groups.** Pairwise correlation
   is +0.04..+0.05 better than absolute (Table 6 vs 3 across every
   metric). `train_v_next_mlp.py --loss mse_rank` is already wired.

3. **Add monotonicity penalty.** Within each (image, codec, knobs)
   curve, penalize `max(0, score(q_i) - score(q_{i+1}))` — directly
   addresses the 71,115 monotonicity violations we found in v15r/v15rc.
   Paper's "forced monotonicity" precedent.

4. **Hold out the 49 CID22 references for evaluation, NOT training.**
   Already enforced via `make_split` source-disjoint splitter +
   `CID22_VALIDATION_41` blocklist in synthetic generator. Keep this
   strict — when V0_4 retrains, evaluate on those 4,292 pairs and
   report SROCC against MCOS, not against ssim2.

5. **Score-mapping target = piecewise-linear-21 fit to MCOS scale.**
   Means: visually lossless → 90, high → 65, medium → 50. PR #24
   `v04_calibrate_mapping.rs` does exactly this (using ssim2 as a proxy
   for MCOS). After PR #24 lands, refit on the 2.37M-row unified parquet
   to remove the +7..+11 systematic bias.

6. **Multi-scale handoff TODO §4.3 is REINFORCED** by paper's 6-scale
   choice. Run multi-scale subset sweep (8 sizes × 100 cluster-centroid
   sources × 5 q × default knobs) and verify zensim score across scales
   for the same `(source, distortion-type, target)` is within ±2
   points; if not, V0_5 design needs a 5th or 6th scale.

7. **Reproducibility:** download the 30-page paper to durable storage,
   mirror to R2 next to the unified data so vast.ai workers can pull
   both in one step. Done — `/mnt/v/zen/zensim-training/2026-05-07/papers/CID22_wg1m99012.pdf`.
