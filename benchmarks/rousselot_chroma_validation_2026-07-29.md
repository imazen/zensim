# Rousselot HDdtb/4Kdtb chroma blind-spot validation — **no measurable chroma-feature necessity; chroma lanes actively INVERT the one pure-chroma ranking axis — a future chroma wave is NOT justified by this data** (2026-07-29)

**VERDICT (registered bands, pre-registered protocol):** on the only public
HDR/WCG image-MOS data with explicit chroma distortions (gamut mismatch,
chroma-targeted Gaussian noise, chroma-Qp compression treatments — Rousselot
HDdtb + 4Kdtb, BT.2020, BVM-X300):

1. **Q1 (decisive, HDdtb chroma-pure n=40): NO measurable chroma-feature
   necessity.** Scene-disjoint (LOSO) ridge probes: full-944 SROCC 0.6489 vs
   **Y-only 0.6220** vs chroma-only 0.3414. Δ(full − Y-only) = **+0.0269**,
   10k paired row-bootstrap p(Δ≤0) = 0.398, CI95 [−0.127, +0.193]
   (scene-cluster: p 0.373, CI [−0.118, +0.274]) — the registered
   "inconclusive at this n" band (real-gap bar was Δ ≥ +0.05 with p ≤ .05;
   hard-negative bar Δ < +0.02, missed by 0.007). Every corroborating
   registered look lands on the null-or-negative side: the K=100 robustness
   extraction flips the sign (Δ = **−0.0464**, p(Δ≤0) = 0.786), the expert-MOS
   secondary read is negative (Δ = −0.0193, p 0.624), and Y-only ≥ full in
   6 of 7 distortion families (L2). A Y-only feature set loses nothing
   measurable — even on pure chroma distortions.
2. **The sharpest measured fact (4Kdtb matched-(scene,Qp) method contrast —
   luma quantization identical, only the chroma treatment varies): the
   chroma features actively ANTI-RANK human judgment.** Mean within-block
   Kendall τ: full-944 **−0.336** vs Y-only −0.037 (≈ blind, as expected)
   vs chroma-only −0.118, score228 −0.097, −RMSE −0.118. Paired block
   bootstrap Δτ(full − Y-only) = **−0.298, CI95 [−0.593, −0.017],
   p(Δτ≥0) = 0.019** — significant harm; reproduced under K=100 (full
   −0.315 vs Y-only +0.063). Mechanism (exploratory rank table): the
   chroma lanes correctly order PHYSICAL chroma error (8-bit-chroma >
   no-cQp-offset > Suppl.15-offset: X-MSE mean within-block rank
   2.94/1.94/1.12) but observers preferred the chroma-damaged variants
   (human mean rank 2.06/2.11/1.83) — the chroma error that the lanes
   measure at these magnitudes (4:2:0-subsampled, high-frequency chroma)
   is largely invisible to humans, the classic missing-chroma-CSF story.
   The lanes are not blind; they are **perceptually mis-weighted**.
3. **The blind-spot hypothesis in its strong form ("zensim can't see chroma
   distortions") is refuted:** single existing chroma lanes reach |SROCC|
   0.83–0.85 on the pure-chroma rows (X-v2 IW_ART 0.849, X-v1 ssim_2nd
   0.845), and the SHIPPING fixed readout (score228 — v1 core incl. X/B
   lanes under SDR-trained weights) hits **0.835** on chroma-pure rows,
   beating every locally-fitted probe. Signal exists and is already
   consumed; what is missing is chroma-visibility conditioning, and at
   head level even that shows no measurable cost here (Y-only matches
   full).
4. **Disposition:** no chroma wave on this evidence. f956..f979 stay FREED
   (`hdr_dmean_commensurability_2026-07-29.md`); the gap-audit chroma
   additions already in-tree measure at-or-below the plain v1/v2 X/B lanes
   univariately (X-append 0.833 ≤ X-v2 0.849; B-append 0.730 ≤ B-v2 0.741;
   **K_XCH is the weakest group of all, 0.716**). If chroma work ever
   revives, the measured target is *visibility weighting* (the 4Kdtb
   inversion — down-weight invisible chroma error, chroma-CSF-shaped), not
   signal coverage — and it needs data where chroma treatment separates
   MOS more strongly than these 96-row sets.

## Protocol / provenance (pre-registered before any number)

`/mnt/v/output/zensim/rousselot-chroma-2026-07-29/PROTOCOL.md` — registered
after dataset-structure inspection only (zero extractions, zero MOS↔feature
contact). Harness: `examples/rousselot_features_extract.rs`, examples-only
commit, **build_commit 73734d8820b46c825aea26f8e4511d50e6a92dc7**
(merge-base-verified on origin/main before extraction). Analysis
`analyze_rousselot.py` + `results.json` in the artifact dir; logs
`~/tmp/rousselot-{extract,analysis*}.log`. One mid-run harness repair
(Expert_MOS repeated in-sheet header row crashed the parser BEFORE any L3
number was computed) — recorded as PROTOCOL Deviations #1; the rerun's
L1/L2/comparator section verified line-identical to run 1 (deterministic,
seed 20260729).

**Data:** HDdtb (EUSIPCO 2018): 8 refs + 96 distorted 944×1080 Radiance
.hdr; families cnoise 24 (chroma Gaussian noise SNR 3/1/0.5) / gamut 16
(709→2020 saturate, 2020→709 desaturate) / hevc_cqp 32 (Suppl.15 chroma-Qp
adaptation) / hevc_nocqp 24; MOS = 15 naive observers (DSIS 0-100,
higher = better), expert sheet (9 columns) used only for the registered L3
secondary. 4Kdtb: 8 refs + 96 distorted 1891×2160; all HEVC Qp 20/27/31/36
× {cqp, nocqp, chroma-8-bit}; 13 expert/sensitized observers. **192/192
pairs resolved, 0 drops, 0 nonfinite pixels**; one xlsx filename slip fixed
by its own row label (Market3-dist_Qp15 → _Qp23, on-disk grid Qp23/31/39/43).
Archives sha256-verified vs the Tower SHA256SUMS sidecar.

**Display model (registered):** nits = raw × **179** (Radiance/pfstools
luminous-efficacy convention — the compressed files of BOTH sets ceiling at
raw 55.75 = RGBE-quantized 10000/179, the 10-bit PQ container maximum),
clamp [0, 1000] cd/m² (BVM-X300 measured peak; the papers' own metric
protocol), declared-HDR streaming route
`compute_folded720_append2_features_hdr(…, HdrEncoding::Linear, csfw OFF)` =
mode 944. BT.2020 primaries fed AS-IS (registered deviation with rationale:
any 2020→709 conversion either hard-clips exactly the chroma differences
under test or pushes negative light into the PU domain; as-is is a uniform
reinterpretation identical for ref/dist — rank statistics only, per set,
never pooled). Registered robustness leg: full recompute at K=100 —
reported above; verdict numbers are K=179 only. Max clamp fraction: HDdtb
16.5% of channel-values on the brightest FireEater variants (fire content
graded ~4,400 cd/m² peak), 4Kdtb 0.7%.

**Feature subsets (registered; formulas + counts in PROTOCOL.md):**
Y-only = 252 lanes (v1/v2/append Y-channel + append2; EXCLUDES the XMASK
lane, whose denominator reads X/B activity), chroma-only = 476 lanes
(v1/v2/append X- and B-channel + the 4 XMASK/K_XCH lanes), dead f156..371 =
216 structural zeros. Probe: LOSO(8 scenes) outer, GroupKFold(4) inner for
λ ∈ {1e-2..100}, ridge on z-scored features, target = raw MOS, scipy
spearmanr — the SI-HDR study machinery with LOSO instead of GroupKFold(5).
λ picks were grid-edge 100 for most folds (recorded; capacity-starved at
n=96, see caveats).

## L1 — the decisive numbers (K=179)

| statistic | full-944 | Y-only (252) | chroma-only (476) | score228 | −RMSE |
|---|--:|--:|--:|--:|--:|
| HDdtb all 96 (LOSO OOF) | +0.8099 | **+0.8190** | +0.5428 | **+0.8841** | +0.4123 |
| **HDdtb chroma-pure 40** | +0.6489 | +0.6220 | +0.3414 | **+0.8354** | +0.5966 |
| 4Kdtb all 96 (LOSO OOF) | +0.6418 | **+0.8082** | +0.6204 | **+0.8282** | +0.5416 |
| 4Kdtb mean block-τ (32 blocks) | **−0.3359** | −0.0370 | −0.1182 | −0.0974 | −0.1182 |

- Δ(full − Y-only), chroma-pure: **+0.0269** — row-boot p(Δ≤0) 0.398
  CI95 [−0.127, +0.193]; scene-boot p 0.373 CI95 [−0.118, +0.274].
  Registered band: **inconclusive at this n** (not the <+0.02 hard-negative
  band; nowhere near the ≥+0.05-with-p≤.05 real-gap band).
- chroma-only vs Y-only on the same rows: Δ = −0.281 (p(Δ≤0) = 0.928) —
  the chroma-lane probe is far WORSE than the Y probe even on pure chroma
  distortions.
- Δτ(full − Y-only), 4Kdtb blocks: **−0.298, CI95 [−0.593, −0.017],
  p(Δτ≥0) = 0.019** — the one significant registered difference in the
  study, and it is chroma-feature HARM.
- HDdtb matched-(scene,Qp) dist-vs-distc sign agreement (24 pairs): full
  10/24, Y-only 13/24, chroma 11/24, score228 11/24 — chance-level for
  everything; the with/without-cQp MOS deltas in HDdtb are small.
- K=100 robustness (registered, non-verdict): chroma-pure full 0.5737 /
  Y-only 0.6201, Δ = −0.0464 (p(Δ≤0) 0.786); block-τ full −0.315 /
  Y-only +0.063. The +0.027 primary point estimate is not mapping-robust;
  the τ inversion is.

## L2 — per-family SROCC (LOSO OOF, K=179)

| family | n | full-944 | Y-only | chroma-only | score228 |
|---|--:|--:|--:|--:|--:|
| hddtb.cnoise | 24 | +0.6061 | **+0.6609** | −0.1643 | **+0.8539** |
| hddtb.gamut | 16 | +0.0530 | +0.2855 | −0.0942 | +0.4018 |
| hddtb.hevc_cqp | 32 | +0.8772 | **+0.9139** | +0.6177 | +0.9084 |
| hddtb.hevc_nocqp | 24 | +0.8374 | **+0.8670** | +0.5278 | +0.8652 |
| 4kdtb.hevc_chroma8b | 32 | +0.6199 | **+0.8817** | +0.5995 | +0.8329 |
| 4kdtb.hevc_cqp | 32 | **+0.7957** | +0.7752 | +0.6958 | +0.7847 |
| 4kdtb.hevc_nocqp | 32 | +0.5290 | +0.7660 | +0.5552 | **+0.8824** |

Y-only ≥ full in 6/7 families; the fitted chroma-only probe is NEGATIVE on
both pure-chroma families. Where chroma features change the full model most
(4kdtb chroma8b/nocqp: full 0.62/0.53 vs Y-only 0.88/0.77), they HURT.
Gamut is near-unpredictable for everything incl. score228 (0.40) — see
caveats: gamut MOS is high and tight (72.5 ± 7.4 on 0-100, per-row CIs
±8-18; EUSIPCO records naive observers found gamut errors "quite hard to
evaluate"), so family-internal ranking is mostly observer noise.

## L3 — expert-MOS secondary (registered, non-verdict)

96/96 expert rows matched on (scene, group, level). Naive↔expert MOS SROCC
**0.9445** (all) / 0.9416 (chroma-pure) — the two observer populations rank
near-identically, closing the "experts would rank chroma damage
differently" escape hatch. Probe numbers vs expert MOS (chroma-pure): full
0.6013 / Y-only 0.6206 / chroma-only 0.2615; Δ(full − Y-only) = −0.0193
(p(Δ≤0) 0.624). Same null.

## Q3 — lane attribution (zero-fit, diagnostic)

HDdtb chroma-pure (n=40), best |SROCC| per group (697 live lanes; top-15 is
ENTIRELY Y-channel, 0.85–0.88, correct polarity):

| group | best lane | \|SROCC\| |
|---|---|--:|
| Y-v2 | v2.s1.Y.HF_GAIN (f494) | 0.876 |
| Y-v1 | v1.s1.Y.edge_art_mean (f55) | 0.868 |
| Y-append | app.s1.Y.CONTRAST_GAIN (f795) | 0.858 |
| X-v2 | v2.s0.X.IW_ART (f389) | 0.849 |
| X-v1 | v1.s0.X.ssim_2nd (f2) | 0.845 |
| X-append | app.s0.X.GMS_DEV2 (f730) | 0.833 |
| append2-Y | app2.s1.Y.BANDVIS_LOSS (f930) | 0.814 |
| B-v2 | v2.s0.B.IW_SSIM (f446) | 0.741 |
| B-v1 | v1.s0.B.ssim_mean (f26) | 0.735 |
| B-append | app.s1.B.MSCN_DIFF_L2 (f811) | 0.730 |
| **K_XCH** | app.s0.Y.XMASK_TRANSDUCER (f737) | **0.716** |

Readings: (a) pure chroma distortions are *rankable from Y lanes alone* —
XYB-Y responds to chroma noise (after 4:2:0 resampling), saturation shifts,
and their scene-level visibility structure; (b) the gap-audit append
additions do not out-rank the plain per-channel lanes they were meant to
extend (X-append < X-v2, B-append < B-v2, K_XCH last); (c) B-channel lanes
are uniformly the weakest per-channel family (0.73-0.74) — consistent with
the yellow-violet acuity limit, not with a missing-signal story. 4Kdtb
all-rows top-15 is likewise all-Y (Qp axis dominates, 0.86-0.88).

## Coverage / caveats

- 96 rows × 8 scenes per set; 40 chroma-pure rows; 16 gamut rows; 32
  τ-blocks of 3. Everything here is small-n; the registered claims are the
  bootstrap-qualified ones only.
- LOSO ridge at n=96 × p≤944 is capacity-starved (λ at grid-edge 100 for
  most folds, recorded): absolute probe SROCCs are FLOORS, and score228
  beating all fitted probes shows how much the frame under-uses the
  features. The full-vs-Y-only COMPARISON shares the frame and stays fair —
  every subset gets the same estimator, data, and folds.
- "Y-only matches full" ≠ "chroma distortions are invisible without chroma
  lanes": it means the Y-projections of these distortions already order
  severity as well as the chroma lanes do, at this data mass.
- The 4Kdtb inversion is a within-block statement (matched scene+Qp); it
  does NOT say zensim mis-ranks Qp levels (it ranks those well, L2) — it
  says the marginal chroma-treatment signal the chroma lanes add is
  anti-perceptual at these error magnitudes.
- Per-row MOS 95% CIs are wide (±2..±20 on 0-100); naive HDdtb n=14 votes
  effective. MOS scales never pooled across sets.
- Display-model conventions (×179, BT.2020-as-is, [0,1000] clamp) are
  registered facts of the harness; the K=100 leg brackets the scale risk.
  The as-is primaries reading renders WCG content slightly desaturated
  relative to what observers saw — a uniform transform, priced as a caveat,
  not corrected post hoc.
- score228's strength here partially reflects that v1-core X/B lanes with
  big-data SDR-JOD weights ARE a chroma consumer — the "existing chroma
  features carry chroma-distortion MOS" question is answered YES at the
  shipping readout level (0.835 chroma-pure), while the marginal-value
  question for the 716 post-228 lanes is answered NO on this data.

## Artifacts

`/mnt/v/output/zensim/rousselot-chroma-2026-07-29/` — PROTOCOL.md,
COMMANDS.md, pairs_manifest.json, {hddtb,k4dtb}_feats_944_k{179,100}.csv
(96×944 each), rousselot_rmse.csv, analyze_rousselot.py, results.json;
Tower mirror `/mnt/tower/output/zensim-rousselot-2026-07-29/`
(sha256-verified). Dataset provenance:
`zenpapers:datasets/Rousselot-HDdtb-4Kdtb.pointer.md` (+ its Validation
RESULT section, landed same-day).
