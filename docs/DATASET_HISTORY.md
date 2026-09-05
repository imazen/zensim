# Zensim Dataset & Experiment History — accurate record

**Purpose.** An honest ledger of *what training data and experiments were tried, what
actually helped, and — where it differs — the **real** reason a thing failed vs the
reason we thought at the time.* Several "verdicts" in this project were later found to
rest on a hidden bug, confound, or invalid measurement. This doc exists so we don't
re-add a dataset a confound already poisoned, re-run a dead-end, or trust a retracted
claim.

**Status flags** on each verdict:
- **STANDS** — verified, still correct.
- **RE-FRAMED** — the finding survived but the *stated reason* was wrong/incomplete.
- **INVALIDATED** — the evidence was a bug/confound; conclusion does not hold as stated.
- **UNVERIFIED** — asserted but never independently confirmed; treat as provisional.

**How this was built (2026-07-15).** Synthesized from a 4-reader survey of `benchmarks/*.md`,
`docs/*.md`, `CLAUDE.md`, the memory store, and `git log`, plus direct verification of the
two most load-bearing claims (the kadid/tid `ssim2_gpu` misjoin — reproduced this session;
the V0_8-leak doc conflict — confirmed in `CLAUDE.md`). **Every row carries a citation so it
can be re-verified, not venerated** (per the project's "docs almost always lie — trace to
source" rule). Where the survey and a shipped doc disagree, that conflict is called out in §4.

---

## 0. The five recurring "actual why" families — READ THIS FIRST

The dominant class of dead-end in this project was **never a bad mechanism — it was an
invalid selection or verdict instrument.** Almost every entry below is an instance of one
of these five. The guard each produced is the scar tissue; §6 collects them.

1. **The ssim2-target selection trap.** The synthetic training corpora use an *ssim2-derived*
   `human_score`. So any verdict scored by *rank-agreement with an ssim2-shaped signal*
   rewards ssim2-shaped bakes and *penalizes deliberately-different ones* — even when the
   different bake is better on human MOS. Recurs ≥4×: the IW-SSIM "falsification" (§3.3),
   the TV/cvvdp_w1 panel reversal (§3.7), the w11 corpus-refit (§3.4), and `cid22_train`
   anti-correlating with real CID22 as a *selection axis*. **Guard:** never pick candidates
   on an ssim2-derived axis; use held-out human MOS + the full panel.
2. **SROCC-only blindness to dynamic range.** SROCC is rank-invariant, so it is *totally
   blind* to a collapsed/clamped dial. V0_5 bakes were "panel-best" while 85% of predictions
   clamped to 0; V39 had SROCC 0.88 with a dial that ran to −128 at *high* quality.
   **Guard:** mandatory **two-panel** eval (rank *and* dial: G1 dynamic-range + monotonicity).
3. **Post-selection / holdout-fishing on small n.** Picking the best-of-N on the same small
   test set and then reporting its p-value. Burned UPIQ-380 (~21 looks; the BHdr "p=0.005"
   became p=0.221 corrected) and nearly mis-read AIC-4 (n=300). **Guard:** Westfall-Young
   family correction; a holdout is never a recipe-search axis.
4. **Join / feature-extraction integrity.** The metric/model was fine; the *data plumbing*
   silently broadcast or zeroed features. Ref-only joins (kadid/tid misjoin), GPU odd-dim
   garbage (dial-grid corruption), empty→0.0 imputation (picker all-zero features).
   **Guard:** `_validate_metric_columns()`, grid quarantine, fail-loud joins, assert
   features non-constant.
5. **Regime / measurement mismatch.** Comparing across mismatched feature regimes or
   drifted code. u8-shell vs PU-linear grams, validate-vs-runtime spline caps, trainer code
   drift, f32 finite-difference floors. **Guard:** `--features-name` regime tags,
   runtime-parity caps, `[training].trainer_commit` pin.

---

## 1. Training-corpus ledger — what was fit, what actually helped

Shipped today (verified in `zensim/src/profile.rs`): **A** = `v47_strict_qat_native_2026-05-27.bin`
(MLP, deprecated-but-compiled); **B** = `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin`
(linear SDR default); **BHdr** = `bhdr_linear_shaped_cvvdpmix_2026-07-12.bin` (linear HDR, sha `7d7f2123`).

| Dataset | What it is | Used to fit | Verdict | Why (stated) | Cite |
|---|---|---|---|---|---|
| **hdr_v3mix** | 7,410 HDR-JXL renditions; target = cvvdp-mix `0.5·clip01(ssim2)+0.5·clip01((JOD−6)/4)` | B cid head (80%), BHdr, B kon head, B winsor corpus; **@944 (hdr944-leg): the Appendix Q phase-1 MLP co-train leg (w 1.2 = 17.0% pair share) + hdr-only linear control (2026-08-05)** | **HELPED — most load-bearing**; @944 co-train: **SDR-free** (full wave-11 SDR panel held) + korshunov-parity, but **narwaria-blind** (all Q candidates significantly behind BHdr there — zenjxl-only supervision suspected; Q.R7) | the cvvdp-mix TARGET is the driver (+0.039 CID22 vs pure ssim2); transfers to SDR linearly. @944 the corpus trains in-domain 0.93+ MLPs whose UPIQ transfer varies 0.05–0.73 across seeds — in-domain val is BLIND to transfer collapse | `profile_b_methodology_2026-07-12.md §2`; `linear_projections_2026-07-03.md` (commit `1b2bdb9b`); `sota944_campaign_2026-08-03.md` Appendix Q/Q.R |
| **safesyn** | 196,086 synthetic tiles (CLIC/CID22/kodak/gb82-sc crops), ssim2-derived target, CID22-49-purged | A (v11→v47), B kon head | **HELPED (foundational)** | base training mass; supplies the BVLS kon head its bulk | `CLAUDE.md` "Safe synthetic dataset"; `DATA_SPLITS.md §3` |
| **cid22_train** (201-ref) | 17,611 rows, **ssim2-anchored** (`human_score`=`ssim2_gpu/100`, NOT MCOS) | A, B kon head | **HELPED (train-legal)** | 201 refs disjoint from the 49-ref holdout; human MCOS never trains | `DATA_SPLITS.md §7.1` |
| **kadid / tid** | 10,125 / 3,000 rows, DMOS/MOS; ~95% analytic (non-compression) distortions | A, B kon head (low weight) | **NEUTRAL / integrity guard** | ssim2 tuned in-sample → never a scoreboard; train==val | `DATA_SPLITS.md §3` (T1) |
| **konjnd-dense** | 20,160 rows (active-mix target) + 1,008-ref val (PJND) | A (modest), B kon head (**≈wash**) | **HELPED (A) / NEUTRAL (B)** | for B the KonJND skill is the **BVLS sign-mask, not the corpus** (`canonkjhdr15`≈`canonhdr15`) | `linear_projections_2026-07-03.md` round-2 |
| **inclusive-winsor corpus** | 10,810 = hdr_v3mix + zenjxl near-lossless SDR sweep; fits B's winsor bounds only | B calibration | **HELPED (user-caught fix)** | frees 245/372 clamped-constant features → near-lossless dial 91.5→96.1 at ~0 MOS cost | `profile_b_methodology §3` (commit `aaa1ecac`) |
| **cvvdp_iwssim_LARGE** | 73,300 rows, `mix_cv40_iw60` target, **300 feats** | PreviewV0_5Balanced (V_22, not a current Profile) | **HELPED (V_22 era)** | +0.068 KADID/+0.087 TID; "marginal value is in content types not in training" | `v22_mix_LARGE_iwssim_methodology_2026-05-18.md` |
| **bigcodec / imazen-26** (canonical-picker) | 2.9–5.7M ssim2-normalized real-codec cells, 414 imazen-26 origins | MLP recipes (w≈0.25, later 0.5); **EXCLUDED from linear B** | **⚠ RE-FRAMED 2026-08-04 — the exclusion was RIGHT and its MLP carve-out was NEVER VALIDATED.** Original: "EXCLUDED from linear — poisons CID22 (0.65–0.76); MLPs absorb it via capacity" | a 372→1 linear head gets pulled to the ssim2 multi-codec regime. **"MLPs absorb it" was tested on CID22 ONLY. Measured 2026-08-04: at ~78% of row mass (bigcodec + its distillation teachers) the 944-era recipe collapses classic-IQA breadth AND KADID at EVERY feature width — 372-width arms on the 944 data score CSIQ 0.681/0.438, LIVE 0.666/0.300, KADID 0.546/0.641 vs B 0.934/0.897/0.809 and winner_dial 0.958/0.960/0.946. MLPs did not absorb it; they inherited it on the axes nobody re-checked.** **⚠ KADID FIGURES CORRECTED 2026-08-04 (§3.20): the ext-lineage tables store KADID's target INVERTED, so every KADID number in this row was published as an unsigned magnitude of a SIGNED quantity. Against KADID's real human MOS the 372-width-on-944-data arms are **−0.546 / −0.641** (anti-correlated), B is **+0.809** and winner_dial **+0.946**. The row's CONCLUSION is unchanged and in fact understated — the 944 recipe does not merely weaken KADID, it INVERTS it. CSIQ/LIVE/CID22 in this row are unaffected (those corpora are correctly oriented on every root).** Why it stayed invisible: KADID/TID were classified "integrity guard, never a scoreboard" (row above), and CSIQ/LIVE did not exist as corpora until `ac787382` (2026-07-18), AFTER the MLP recipes were set — so no gate in force at the time could fail. | `linear_projections_2026-07-03.md` Falsified #1; commit `c7900d53`; falsification: `sota944_campaign_2026-08-03.md` width-discriminator (k=2/arm — seed spread too wide to rank widths; the breadth/KADID collapse is robust to that limitation) |
| **KADIS-700k** | 700k analytic-distortion cells on 140k Pixabay refs; GPU-metric targets, no human labels | negative-region calibration; the §8.32-8.33 MLP-neg prototype (not shipped) | **EXCLUDED for shipped fit; diagnostic/negatives only** | cvvdp-scalar target falsified (V41); analytic + more-photographic bias | `DATA_SPLITS.md §2b`; `project_profile_b_hdr.md §8.31-8.33` |
| **kadis-hdr** | 11,400 PQ-domain synthetic HDR cells, 1,140 imazen-26 refs | BHdr breadth retrain | **EXCLUDED — falsified on UPIQ** | synthetic HDR breadth does not transfer to human-MOS HDR (linear family) | `bhdr_improvement_split_lineage §8.8-8.15` |
| **CID22-49 / AIC-3 / AIC-4 / UPIQ-HDR / JPEG-AI-SDR25** | human-MOS holdouts | **HOLDOUT-ONLY** | **EXCLUDED (sacred)** | the only held-out human generalization gates; UPIQ-380 now BURNED (~21 looks) | `CLAUDE.md` "CID22 is VALIDATION-ONLY"; `DATA_SPLITS.md §3` |

**Superseded / unverified-ship:** hdr_v3 (pure-ssim2, prior BHdr `anchored2`); e1-fill/zenjpeg-420-e1 (V0_7 line, built ~85%, **no promoted-bake trail** → UNVERIFIED); KonFiG-IQA (v53 group, **no shipped-bake trail** → UNVERIFIED); cvvdp-only LARGE (falsified, craters KonJND); dssim co-train (falsified, see §2). Pre-`VALIDATION-ONLY`-era weights (Nelder-Mead `ae28074`, CMA-ES) trained on {KADIK, CID22, TID} *including CID22* — any CID22 SROCC from that era is **contaminated**, archaeology only.

**AIC-HDR2025: UNOBTAINABLE (user ruling 2026-08-05)** — the dataset was never publicly released (README-only clone at `/mnt/v/datasets/aic-hdr2025/`) and the authors are unresponsive; STOP live-checking `github.com/jpeg-aic/AIC-HDR2025`. Removed from the HDR anchor plans; it can never enter train or eval.

---

## 2. What didn't work — falsification ledger (verdict strength matters)

Ranked by consequence. **Verdict strength** distinguishes a rigorous multi-stat/multi-seed
falsification from a single-run or SROCC-only call (the latter are re-flagged in §4).

| Approach | Why it failed (stated) | Strength | Re-opened? | Cite |
|---|---|---|---|---|
| **CVVDP-scalar-target** (predict CVVDP's pooled JOD) | "emulating CVVDP's OUTPUT ≠ having its CSF mechanism"; strictly worse ranker (CID22 0.66/0.70) | ⚠ **RE-FRAMED 2026-07-15** — a TARGET-SHAPE confound, not a cvvdp limit (§3.17). RAW `cvvdp_score` trains fine (CID22 0.85); only `cvvdp_log_norm`'s near-lossless-tail expansion craters it (0.58). | RE-OPENED — raw-scalar cvvdp is a viable HF lever | `benchmarks/…§8.36`; `v47_cvvdp_target_FALSIFIED_2026-05-27.md` |
| **IW-SSIM direction** | "wins KADID+TID, craters CID22 (FRIQUEE transfer)" | ⚠ MIXED — the CID22 crater is the ssim2-SROCC trap (§0.1); IW won TID best-ever (Z-RMSE 0.231) | created "SROCC-only BANNED"; single-MLP high-k re-confirmed on full panel | `CLAUDE.md` "SROCC-only BANNED"; `falsification_reeval_results_2026-05-15.md` |
| **Su-2023 distortion-manifold pretrain** | wins KADID+TID, loses held-out CID22 −0.027 (synth→authentic transfer) | multi-stat, multi-corpus | no | `docs/v0_20_path_evaluation_2026-05-14.md §B` |
| **bigcodec mass in a linear fit** | poisons CID22 (0.65–0.76 at ≥0.25 w); MLPs absorb it | deterministic refits | durable rule "keep bigcodec OUT of linear" | `linear_projections_2026-07-03.md` |
| **BHdr kadis-hdr breadth retrain** | synthetic HDR breadth ⊄ human-MOS HDR | ⚠→ RIGOROUS *after* a regime confound was fixed (§3.9) | withdrawn then re-confirmed | `bhdr_improvement_split_lineage §8.9-8.15` |
| **BHdr cvvdp-mix "significant UPIQ win"** | promotion basis was POST-SELECTION (p=0.005→0.221 corrected) | RIGOROUS retraction | shipped anyway; UPIQ-380 burned | `bhdr_improvement_split_lineage §7` |
| **hdranch3 BHdr ramp-proxy** | optimized a proxy, never ran UPIQ; UPIQ 0.606 vs 0.7536 | RIGOROUS multi-stratum | dead; installed "UPIQ-panel-before-ship" | `project_bhdr_hdranch3_falsified.md` |
| **G5 KonJND-HF ≥0.70** (agg head + regime ensemble) | Pareto limit — clearing 0.70 craters CID22; regimes overlap in feature space | falsified ×2 architectures | dead; needs better HF *representation* | `CLAUDE.md` V39-learning #9 |
| **dssim co-training** | regressed CID22 0.04–0.07 | ⚠ single-cycle, **confounded** (61% was recipe drift, 39% dssim) | retry never funded (bakes stuck ZNPR v2) | `cycle_7_dssim_outcomes_2026-05-12.md` |
| **Dynamic-range-floor regularizer (V40)** | overshoots >100 | ⚠ weak/single-config | no | `CLAUDE.md` V39-learning #7 |
| **Axioms-only residual metric** | A3 (no inversions) not gained; degenerate range (CID22 0.50-0.62) | 2 parameterizations, full panel | no | `axioms_only_residual_falsified_2026-05-27.md` |
| **Cascade / NNLS-all-pinned ensembles** | dominated by direct convex blends | deterministic 44/44 | no | `linear_projections_2026-07-03.md` |
| **D1/D2/D3 input-shaping** | V_20 IS is a B3 specialist (−0.014 aggregate); shaping is intrinsic | train+eval per variant | D2 shipped as niche PreviewV0_4 | `CLAUDE.md` "V_20 learnings" |
| **Recipe knobs** (low/mid-q boost, cosine LR, h=64/256, NiN-off, per-group std, P²) | none broke the V_18 CID22 ceiling | mixed (cycle-9/12 re-confirmed) | ~120 cycle-7-13 bakes are ZNPR v2, retry deprioritized | `falsification_reeval_results_2026-05-15.md` |

---

## 3. The confounds — "we thought X, the actual why was Y"

The most important section. Each: **thought-why → actual-why**, when discovered, status,
shipped-bake impact. All shipped bakes were verified *unaffected* except where noted.

### Tier 1 — highest consequence

**3.20 The ext-lineage KADID TARGET IS INVERTED — `(5−dmos)/4` instead of `(dmos−1)/4`
(discovered 2026-08-04, Tier 1).**
*Thought:* KADID SROCC magnitudes were comparable across the 372/canonical era and the
720/924/944 ext era, so "era 0.946 → 944 0.423" read as a competence regression.
*Actual:* the two lineages store **opposite orientations of the same label**, and every
KADID number ever published from an ext root is **sign-flipped**.
`scripts/canonical_corpus/build_fr_corpus_pairs.py` `build_kadid()` (line 113) emits
`human_score = (5 − dmos)/4` — the standard invert-a-DMOS reflex that CSIQ (`1−DMOS`) and
LIVE (`1−dmos_new/100`) correctly need — but **KADID's `dmos` column is a MOS in
disguise**: raw crowdsourced DCR (349,800 ratings, `raw_crowdsource_data.csv`) falls
4.0789 → 2.0072 across severity levels 1–5, so it was already quality-oriented and the
flip inverts it. The canonical lineage (`build_canonical_parquets.py:288`,
`fix_kadid_tid_build_pairs.py:15`) uses the correct `(dmos−1)/4`. Both residuals are
**exactly 0.0** — two transforms, not drift.
**Consequences.** (i) Every `ext720`/`ext924`/`ext944` KADID SROCC is the negative of the
true-quality value; **110 of 188 board bakes are anti-correlated with KADID's real human
MOS** while the board renders all 188 as positive magnitudes. (ii) Every 944-era model
**trained** on the flipped column (`--group kadid:…ext_kadid.parquet`), so its KADID
inversion is real and inherited — dose-response over 111 fullevals: train weight 0.50 →
mean −0.457, weight 1.50 → mean −0.925. (iii) The wave-8 **E1 gate ("KADID ≥ 0.70")** was
passed by the three most-inverted arms (−0.93) and failed by `W8C_s3101`, the only
wave-8 arm whose KADID was correctly oriented (+0.358). (iv) The pre-ext era models are
**unaffected in behaviour** — winner_dial is +0.9464 and shipped **B** +0.8201/+0.8085,
positive on **25/25** KADID distortion types; only their reported sign was wrong.
**TID is CLEAN on every root** (verified against the same raw ratings, +0.9168, n=960);
so are CSIQ and LIVE (their natives genuinely are distortion-oriented).
**Status: eval reporting INVALIDATED for ext-lineage KADID; the 944 models' KADID
behaviour is a real defect requiring a retrain on a fixed table if KADID competence is
wanted. Shipped bakes (B/BHdr/A) unaffected — all pre-date the ext lineage and trained on
the correct column.** Cite: `benchmarks/sota944_campaign_2026-08-03.md` REGISTERED
APPENDIX F + F.R1..F.R8; `benchmarks/wave9/kadid_orientation_2026-08-04.md` (which
measured the orientation and flagged the anomaly one day earlier);
`benchmarks/eval_annotations.json` entries `kadid-ext-root-inverted` +
`kadid-ext-trained-inverted-model`.

**3.1 kadid/tid canonical `ssim2_gpu` = ref-vs-ref MISJOIN; `iwssim` = human-score copy.**
*Thought:* KADID/TID carried real IW-SSIM + SSIMULACRA2 columns → multi-target training was
legitimate. *Actual:* two committed bugs — `iwssim` was a literal copy of `human_score` (a
val-only mock whose "mock" qualifier was stripped upstream → training on the eval label);
`ssim2_gpu` was joined on `ref_basename` alone (codec/q keys dropped) → every ref's ~125
distortions averaged to one ≈99.99 value → corr ≈0.01 with MOS. **The metric was sound; the
Python join collapsed it.** Discovered 2026-05-25. **Status: RE-FRAMED/fixed** — re-joined
positionally (post-fix real SROCC KADID ssim2 0.013→0.813); `_fixed_2026-05-25.parquet`
siblings + a `_validate_metric_columns()` guard. **Shipped bakes: NONE affected** (all train
on `human_score`/real cvvdp). **INVALID:** ~10 multi-target experiments with kadid/tid
train_weight>0 on the contaminated column. *Verified this session* — training on the current
canonical `ssim2_gpu` ranks kadid **backwards** (val SROCC −0.07..−0.13). Cite:
`DATA_INTEGRITY_kadid_tid_metric_columns_2026-05-25.md`, `DATA_INTEGRITY_root_cause_2026-05-25.md`.

**3.2 dHash d≤16 "contamination" cleanups = mostly FALSE POSITIVES.**
*Thought:* V0_8's CID22 0.8948 was inflated by 361 near-dup training sources leaking 22/49
CID22 refs (drove a 30.6 GiB purge + V0_15 retrain); and V0_18's CID22 was inflated by KADID
overlap at d≤16 (drove the V0_18→V0_19 ship). *Actual:* **d≤16 is the LOOSE screening
threshold, not a contamination cutoff.** User montage review confirmed NONE of the flagged
matches are the same image (flat-region/UI/composition false positives); at d≤10 CID22↔KADID
and CID22↔TID have ZERO overlap. Discovered 2026-05-14. **Status: V0_19 REVERTED** (V0_18
restored), "V0_18 inflated" RETRACTED. **⚠ OPEN:** the 2026-05-12 V0_8 361-source purge used
the *same* loose d≤16 threshold, was flagged "re-audit at d≤10," and **never was** — so the
"V0_8 was inflated by leakage" claim is itself **UNVERIFIED** under the same false-positive
caveat. See §4.1. Cite: `dhash_threshold_revert_2026-05-14.md`, `v0_19_REVERTED_2026-05-14.md`.

**3.3 IW-SSIM "falsified on CID22 SROCC" = the ssim2-target selection trap (§0.1).**
*Thought:* IW bakes are falsified — CID22 SROCC craters to 0.4632. *Actual:* the corpus's
`human_score` is ssim2-derived, so CID22 SROCC rewards ssim2-shape and penalizes IW's
deliberately-different surface; the same bakes WIN TID human MOS + PWRC + Z-RMSE (best TID
ever). Discovered 2026-05-15. **Status: RE-FRAMED** — created the project-wide **SROCC-only
BANNED** rule; all prior SROCC-only "falsified" verdicts marked provisional. Cite:
`CLAUDE.md:408-488`.

**3.4 2026-05-29 DIAL GRID corruption — masked/IW features were GPU odd-dim garbage.**
*Thought:* shipped B scored −80/−81 on a "webp trio" → B-specific OOD content blindness →
proposed corpus-refit / knob-floor. *Actual:* the −80s came from the **grid's stored
features**, not the model — on 9/115 ladders (8/24 webp) the masked/IW features (f228..f371)
were extraction garbage, **bit-constant across each ladder's 40 q-values** (a `zensim-gpu`
odd-dim pyramid pathology emitting non-NaN garbage). Fresh CPU extract → sane + ordered; the
webp knob-blocker **never existed** (quarantined grid p10 83.7 vs 9.4). Discovered 2026-07-05.
**Status: grid QUARANTINED** (`dial_grid_372col_2026-05-29_quarantined.parquet`) — any dial
number on those 9 ladders since 2026-05-29 by ANY bake is garbage-input. The corpus-refit
"fix" was run anyway and independently falsified (ssim2-target trap again). **Shipped bakes:
none affected.** Cite: `MEMORY.md:30`, `provenance_best_results_2026-07-04.md §w11` (commit `ae4209a8`).

### Tier 2 — significant

**3.5 V0_5 "panel-best" bakes hid a BROKEN DIAL (SROCC-only, §0.2).** Distance-shaped raw
(≈[−30,30]) → runtime `clamp(0,100)` pinned 84.6% of predictions to 0; SROCC blind to it.
**Status: fixed** via PCHIP spline retrofit (rank-invariant) + the mandatory two-panel
scorecard. Cite: `dial_bug_audit_2026-05-20.md`.

**3.6 BHdr "UPIQ +0.0223 significant p=0.005" = POST-SELECTION (§0.3).** λ picked best-of-7
on the same 380 UPIQ pairs; Westfall-Young maxT → **p=0.221 NS**, "6/7 corpora" → 4W/2L/1-ns.
Discovered same day as ship (`f5704efb`). **Status: improvement NOT established (non-inferior
only); ship kept as-is.** The λ-robust out-of-domain SDR gains (CID22/TID/KonJND) still stand.
Cite: `project_profile_b_hdr.md §7`.

**3.7 TV/KADIS "A is Pareto-optimal" = 3 data misreads.** (1) TV pairs were POISONED — signed
U-shaped distortion types 7/18/25 excluded from the safety grid but not the TV pairs → 6.68%
taught wrong order → every "dirty-TV KADID crater" is contaminated. (2) the oracle monotonicity
ceiling is 0.980 (cvvdp's own step-inversion on real reversals) → chasing 0.99 flattened real
reversals. (3) full panel *reverses* the aggregate-SROCC story (cvvdp_w1 beats A on 4/6). 
**Status: clean pairs rebuilt; A stays (CID22 gap real ~4 SEM).** Cite: `project_tv_monotonicity_pareto.md`.

**3.8 AIC-3 "0.80 vs 0.96 CVVDP gap" = 5-image-subset ARTIFACT.** CVVDP's 0.96 was 5 images;
on the full 600-pair set CVVDP is 0.79 pooled / 0.93 per-ref and our bake is 0.9475 per-ref
(ABOVE raw CVVDP). **Status: SUPERSEDES "need CSF features to close the CVVDP gap"** — prevented
a wasted feature-engineering investment. Cite: `aic3_cvvdp_feature_spike_2026-05-25.md`.

**3.9 kadis-hdr breadth falsification — FIRST confounded (regime), THEN clean (§0.5).** §8.12
failed the UPIQ gate → "breadth falsified," but the kadis features were u8-shell while the
family is PU-linear (a "regime chimera"). §8.13 WITHDREW it; re-extracted PU-linear, re-ran →
§8.15 clean re-falsification (fails both UPIQ strata). **Status: conclusion SURVIVED after the
confound was removed; the original numbers were invalid.** Regime-consistency now enforced.
Cite: `project_profile_b_hdr.md:370-390`.

**3.10 B near-lossless "intrinsic to linearity (falsified 3 ways)" → MISCALIBRATED WINSOR.**
*Thought:* B's linear projection can't rank near-lossless; falsified 3 ways → "A is the nl knob."
*Actual:* B's shipped winsor bounds turned 245/372 features CONSTANT on nl content — self-inflicted.
Clean bounds: nl per-img SROCC 0.286→0.886, CID22 +0.049, at ~0 MOS cost. **Status: RE-FRAMED,
fixed + shipped (inclusive-winsor B).** NUANCE: a later decomposition (`98e9f395`) found the
dominant cause was ~80% feature-vanishing + ~15-20% winsor, so the "winsor is THE cause" framing
was itself partially walked back — but the fix stands. Cite: commits `a6edbced`→`aaa1ecac`→`98e9f395`.

**3.17 CVVDP-scalar "dead end" = the `cvvdp_log_norm` TARGET SHAPE, not cvvdp (2026-07-15).**
*Thought:* training toward CVVDP's scalar output is a dead end (V41: CID22 0.66 vs ssim2 0.88) —
"emulating CVVDP's output ≠ its CSF mechanism." *Actual:* on safesyn cvvdp is 100% present,
learnable (feat→cvvdp 0.987), and agrees with ssim2 (0.984). Same de-poisoned pipeline: **raw
`cvvdp_score` → CID22 0.85** (fine), **`cvvdp_log_norm` → CID22 0.58** (craters). log_norm is
rank-identical to score (SROCC 1.0000) but **exponentially expands the near-lossless tail** (37%
of pairs in cvvdp [9.9,10] → log_norm [27.75,100], std 20), so MSE loss over-weights the top and
under-fits the rest. **Status: RE-FRAMED** — the blanket "cvvdp_* scalar is bad" is too broad; raw
`cvvdp_score` is viable, and per Mohammadi 2025 (cvvdp best in HF/near-lossless) it's the indicated
lever for the HF regime ssim2 can't rank. **Guard: never MSE-regress a log-expanded target; use
raw or a rank loss.** Shipped bakes unaffected (all ssim2-trained). Caveat: V41's exact target
column not re-confirmed from its log; the de-poisoned probe isolates the shape effect regardless.
Cite: `§8.36`.

### Tier 3 — contained (caught at/near ship)

- **3.11 V39 output-spline upper-extrapolation** — validate-side extrapolated uncapped above the
  top knot while runtime caps ≤100 → dial-p95 artifacts of 321-504; "bit-exact" claim false above
  the knot. **RESOLVED** (`5d4978db`, capped for parity). Validate-side only; no bake affected.
- **3.12 V39 "universally better than V0_3" = actually 5/6** — V0_3 significantly wins AIC-4
  (Δ+0.023, p=0.001, invisible under single-SROCC CI). **RE-FRAMED**; V39 shipped anyway (right
  call for the 5 compression holdouts). Cite: `v5_vs_v03_comparison_2026-05-25.md`.
- **3.13 Profile-A "reproduction regression" = TRAINER CODE DRIFT.** Recipe re-run on current
  main gave a collapsed 57KB bake; at the pinned tree `e9442678` it reproduces byte-identically.
  **RESOLVED** — landed a `[training].trainer_commit` gate. No bake affected.
- **3.14 konjnd-agg 2-layer gradient "bug" = MALFORMED FD TEST.** f32 forward floored the central
  difference (ε=1e-6) + a pure-relative gate unbounded as grad→0. Gradients were correct.
  **RESOLVED** (`fix #35`, ε=1e-2 + atol/rtol gradcheck). A *separate* real ~3% pool-head issue
  found + fixed same investigation; no bake affected. Cite: `CLAUDE.md:20-38`.
- **3.15 zenjpeg picker "SROCC 0.341" = trained on the DIAL ALONE.** All 108 `feat_*` were 0.0
  (empty→0.0 join imputation); the picker never saw image content. **RE-FRAMED** as the zq-only
  baseline; re-joined fail-loud. (zenjpeg codec, not core zensim.) Cite: `project_picker_retrain_zenpredict_caps.md`.
- **3.16 konjnd val "negative polarity bug" = STRUCTURAL anti-correlation + naming collision.**
  val/konjnd SROCC is structurally negative (at-PJND pairs) → `|SROCC|` is correct; and
  `human_score` means TWO quantities (train active-mix [0,1] vs val mean-PJND [22,70]) → a default
  `--target-column human_score` fits the wrong one. Interpretation trap, not a code bug. Cite:
  `feedback_konjnd_human_score_two_columns.md`.
- **3.18 kadid/tid `iwssim`/`ssim2_gpu` leaked/misjoined in canonical — RESOLVED 2026-07-15.** The
  V39-#8 bug — `iwssim` = a byte-identical copy of `human_score` (target leak), `ssim2_gpu` =
  ref-vs-ref misjoin — was documented as "fixed in `*_fixed_2026-05-25.parquet` siblings" but those
  were never promoted; verified still-live 2026-07-15 (`iwssim==human_score` max|Δ|=0.00e+00; `ssim2`
  pinned 93.5/100/100). **FIXED by promoting the verified siblings** (`promote_fixed_kadid_tid_2026-07-15.py`,
  user directive): `canonical-2026-05-21/train/{kadid,tid}.parquet` now carry real iwssim
  (SROCC-vs-MOS +0.850/+0.779) + real ssim2 (spans [−367,100]/[−96,90], SROCC +0.813/+0.846); all
  376 preserved cols (372 features + human_score/cvvdp_*/pjnd_target) byte-identical → **zero feature
  drift**. Corrupt originals preserved at `<c>.CORRUPT-v39bug.pre-2026-07-15.bak.parquet` (+ R2
  `_archive/` + Tower). Manifest entries [5]/[9] updated (new sha256). R2 + Tower re-synced. Shipped
  bakes were always SAFE (trained on `human_score`, never kadid/tid iwssim). Cite:
  `benchmarks/column_audit_2026-07-15.md`, `…§8.37 A`, `scripts/canonical_corpus/fix_kadid_tid_apply_scores.py`.
  **INDEPENDENTLY CONFIRMED 2026-07-15 (same day, different route) — BOTH corpora:** from-scratch GPU
  re-extractions (`zenmetrics batch --metric ssim2-gpu`, no reference to the parquet) reproduce the
  promoted columns exactly — **KADID +0.8133** vs documented +0.813 (10,125 pairs, range
  **[−367.2, 100.0]** vs documented [−367,100]) and **TID +0.8460** vs documented +0.846 (3,000 pairs).
  On KADID, row-for-row: **SROCC(fresh, canonical) = 1.000000**, mean |Δ| **1.4e-5**, 10,062/10,125 rows
  identical to <1e-4 (the rest ≤0.001 = f32/GPU noise). The promotion is correct, not merely
  self-consistent. Byproduct worth knowing: **kadid/tid ssim2 + cvvdp + iwssim already exist in the
  canonical train parquets** — recomputing them is redundant; only **butteraugli** is genuinely absent
  from the canonical schema. Fresh reference-metric panel (for the dashboard, `refmetrics/`):
  KADID ssim2 +0.8133 / cvvdp +0.8339 / butteraugli↓ +0.5431; TID ssim2 +0.8460 / cvvdp +0.8531 /
  butteraugli↓ +0.6622 — butteraugli is weakest on both, as expected for a compression-tuned metric on
  KADID/TID's ~95% non-compression distortions (a live confirmation of the §2 "KADID/TID are integrity
  guards, not compression signal" framing). Gotcha for re-runs: TID's `reference_images/*.BMP` are not
  decodable by zenmetrics — use `reference_images_png/` (pixel-identical, max|Δ|=0).
- **3.19 IW/masked HF-moment features explode on non-photographic content — unbounded energy +
  a `1/n`-vs-`Σw` normalization bug (2026-07-15).** bigcodec IW/masked reached 5.8e6 (vs photographic
  p99.9=0.48) — traced to (a) `iw_art4`/`iw_det4` being **unbounded, un-per-image-normalized** edge
  energies in XYB (sharp synthetic edges → 7 orders above photo; the masked block blows up
  identically → energy, not weight, is the primary driver), and (b) a genuine secondary bug: the
  extractor normalizes IW moments by `1/n` (`streaming.rs:424`) while the reference `iw_pool.rs:399`
  uses `1/Σw` (`shipped = ref·mean_w^0.25`, ~1.5–2× inflation on high-activity). Not a decode
  corruption (`o_9292.png` is valid in-range). Not a pixel-ZERO-TOLERANCE issue (streaming↔full-image
  agree; both diverge from the unused reference). Remedy: winsor guard (shipped, §8.35) is load-bearing
  and sufficient; the `Σw` switch + per-image energy normalization is a scheduled full-retrain fix.
  Cite: `…§8.37 B`.
  > **⟳ CORRECTED + QUANTIFIED 2026-07-16 (`benchmarks/ssim_moment_explosion_2026-07-16.md`).**
  > Two of this row's claims are wrong; the headline 5.8e6 is right. (i) The
  > exploding features are **`iw_ssim_4th`/`masked_ssim_4th`** (the SSIM-map
  > higher moments, XYB **ch2**, finest scale), NOT `iw_art4`/`iw_det4` — the
  > edge features measured 0.02–0.09 on the shipped extractor. (ii) The `1/n`-
  > vs-`Σw` weight bug is NOT even a "secondary" driver: the shipped weight is
  > `1+4a` and `mean_w` spans only **1.03–1.27×** (measured, full instrument),
  > so fixing it does ~nothing. The ACTUAL cause is the per-pixel SSIM
  > `d = (1 − num_m·num_s/denom_s)·mask` having a `.max(0)` floor but **no upper
  > cap**, and `num_m = 1 − (mu1−mu2)²` having **no C1** → unbounded-negative on
  > high-magnitude chroma → `d` to millions, L4-amplified (worst row: 4th/mean =
  > 630×). Denominator-cancellation was tested and FALSIFIED (f32 floors it at
  > ~C2, 1.2×). Fix = add a C1 to the luminance term (principled) or cap `d`
  > (cheap); both need a re-extract + full panel + sign-off. The winsor guard
  > clamps the symptom; it is why B ships.
- **3.20 The `mix_cv*` (cvvdp×iwssim) target columns are POISON; raw cvvdp is ceiling-saturated
  (2026-07-15, completes 3.17).** `mix_cv50_iw50` → CID22 0.705 (poison): it inherits cvvdp's
  near-lossless saturation AND is log-expanded (safesyn tail 2.06), and is absent on cid22_train
  (0% cov). Raw `cvvdp_score` is SATURATED on codec pairs (safesyn 37%+ in [9.8,10], tail≈0.0004) →
  MSE can't discriminate the top → 0.849; log_norm re-expands it → 0.597. cvvdp is damned as an MSE
  target either way, but rank-fine (SROCC-to-raw 1.0) → use rank/pairwise on raw cvvdp for the HF
  band only. `ssim2_log_norm` (tail 0.04) trains fine (0.880) → the confound is saturation-specific,
  not "log_norm." Cite: `benchmarks/column_audit_2026-07-15.md`, `…§8.37 A`.

- **3.21 raw-cvvdp-rank auxiliary = an AIC-3 lever, NOT a G5/KonJND lever — MEASURED, FALSIFIED
  (2026-07-15).** The §3.17 reframe left open "raw cvvdp is the indicated lever for the HF regime
  ssim2 can't rank (G5)." Tested directly (`cvvdp_hf_probe.py` — deleted 2026-07-29 with the
  falsified campaign, in git history; `ssim2-MSE + lambda*rank(raw cvvdp)`,
  lambda 0->0.6): AIC-3 |SROCC| rises monotonically (+0.005 to +0.017) but **KonJND |SROCC| FALLS
  monotonically (-0.024 to -0.101)** with a small CID22 cost. The two HF holdouts pull opposite —
  cvvdp saturates (~10) where KonJND needs fine PJND discrimination, so its rank signal orders the
  mid-band and de-aligns from PJND. G5 (KonJND floor) is NOT closed by cvvdp supervision — consistent
  with the standing "G5 needs HF *representation*, not supervision" limit. The reframe stands for
  AIC-3/mid-band only. Untested: a separate gated HF head. Cite: `bhdr_improvement_split_lineage §8.38`.

- **3.22 JXL near-lossless encoder bug — training corpora NEVER contaminated; 33 eval-grid cells
  were — AUDITED + PURGED (2026-07-15).** *Thought-why:* "the jxl encoder had a bug at the highest
  qualities and we never cleaned the parquets — there should be replacement rows somewhere," implying
  a training-data contamination needing a purge + re-join. *Actual-why:* **the training corpora never
  sampled the broken zone at all.** The bug (jxl-encoder, two layers: quantized DC stored as `i16`
  saturated at fine distances → corrupt encode at source, content-dependent on `|DC|>32767`; then the
  header lied `modular_16bit_buffer_sufficient=true` so conformant decoders truncated the widened DC
  and desynced the DC ANS stream) only fires at **butteraugli distance ≤0.02** (≈JXL-native quality
  ≥99.7 — the "0.3 quality points" phrasing; `(100−q)/10 = d`). **Distance ≥0.03 is byte-identical /
  hash-proven at EVERY date**, so it needs no date bound and no re-encode. Fixed `008499e1` (i32 DC +
  interim 0.03 floor, zenjxl#18) → `eeb52735` 2026-07-06T06:09Z (`force_modular_32bit`, floor removed,
  jxl-encoder#94); re-verified 2026-07-14 to d=0.001 across jxl-rs/jxl-oxide/zenjxl (`a0f7e870`, CI
  green) — nothing remains broken. MEASURED clean, three independent ways: `safesyn` 26,362
  `zenjxl-e7` rows q5–q100, q=100 → ssim2 med **95.13** / butteraugli med **0.229** (monotone across
  all q, never below the 0.03 floor); `cvvdp_iwssim_LARGE` zenjxl distances are only **{0.5,1,2,5}**
  (16–250× above the boundary; LARGE stayed 73,300 rows); score sidecars JXL q=10..90 → d≥0.5. Also
  verified clean by direct R2/duckdb query: `canonical-picker-2026-06-27/zenjxl_lossy` +
  `canonical-picker-2026-07-01-zensimA/zenjxl_lossy` (q∈[5,90], generic-quality-only — structurally
  never resolves to native d<0.03), `zenjxl_lossy_hqfill_2026-07-01.parquet` (62,958 rows, d∈[0.05,1.3]),
  `hqfill_7metric_sidecar_2026-07-02.parquet` (62,173 rows, d∈[0.05,1.3]).
  **⚠ CORRECTION (2026-07-15, same day):** an earlier revision of this entry said LARGE's 2026-05-18
  "fresh jxl" 8-distance sweep "was PLANNED but BLOCKED and never landed (`c17447f5`)" — that was read
  off the commit message and is **WRONG**. Direct R2 verification (`aws s3 ls --recursive --summarize`
  on `s3://zentrain/multi-codec-2026-05-18/omni/`) shows the sweep **COMPLETED**: 112/112 objects dated
  2026-05-19, 24,800 cells; its 6,400 zenjxl rows span distance **[0.1, 10.0]**. It was never *merged*
  into canonical LARGE — "blocked" described the merge, not the sweep. The clean verdict is unchanged
  (clean either way, by two independent routes), but do not repeat the "never landed" claim.
  **Near-miss worth knowing:** `jxl-dense-20260530` (2k k-means sources × 44-distance ladder **starting
  at 0.025**) is the one sweep that WOULD have mass-produced contaminated rows — **confirmed never
  launched** (`s3://coefficient/jobs/jxl-dense-20260530/` holds only the 2 prep files from 2026-05-30;
  no `chunks/`, `done/`, `features/`, `variants/` ever appeared — blocked on red sweep-image CI,
  `9d8f73a5`, never resumed).
  **Shipped bakes are UNAFFECTED — no purge of training data was
  needed or done.** What *was* contaminated: `dial_grid_372col_2026-05-29` (eval-only) carries 33 JXL
  cells at d=0.025, built ~5 weeks pre-fix — mean feature-L2 **4.011** / max|feat| **59.29** vs the
  healthy d=0.05..0.35 ceiling **1.56** (L2 0.109→0.246), a **37× distortion explosion at the LOWEST
  distance**, backwards from the monotone trend; 4 of 33 unambiguously broken, all high-DC graphic
  content exactly as the DC>i16 mechanism predicts. Purged into
  `dial_grid_372col_2026-05-29_quarantined_v2.parquet` (4,424 rows, sha256 `6546c43e…`, R2
  `s3://zentrain/eval-grids/`), which drops both this and the §w11 webp garbage; originals preserved.
  *Caveat:* LARGE's extreme feature values are the **separate** §3.19 IW-explosion on `gen-chart`
  content — NOT this bug; do not conflate. Cite:
  `benchmarks/jxl_nearlossless_contamination_2026-07-15.md`, `benchmarks/jxl_nearlossless_dial_2026-07-05.md`.

- **3.23 konjnd-dense as a training corpus = a KonJND MEMORIZATION lever, NOT a G5 lever — MEASURED,
  FALSIFIED (2026-07-15).** *Thought-why:* "we have an HF/near-threshold corpus (konjnd-dense, 20,160
  rows) sitting unused in canonical — adding it to the blend should close G5 (the KonJND ≥0.70 floor)."
  *Actual-why:* it lifts KonJND only by turning KonJND into **train==val**. Measured on the honest
  blend (`{safesyn:1, bigcodec:1.5, kadis:0.3}`, ZERO human eval-corpus data), sweeping
  konjnd-dense weight 0→0.5→1→2 (`blend_search.py --round 5`, seeds 1,7,13,17,23, deterministic —
  reproduced byte-for-byte across two runs): **KonJND |SROCC| rises 0.5143 → 0.5411 → 0.5467** (+0.027
  / +0.032) **while AIC-3 — the one HONEST JND holdout — falls monotonically 0.7908 → 0.7858 → 0.7840**
  (−0.005 / −0.007), CID22 flat-to-down (0.8850 → 0.8854 → 0.8827). So the KonJND gain does not
  transfer to held-out JND: it is memorization of the corpus it now trains on. Even at its best, KonJND
  0.5467 is far under the 0.70 G5 floor.
  **Symmetry with §3.21 — two opposite supervisions, same verdict:** §3.21 pushed raw-cvvdp rank and
  got AIC-3 UP / KonJND DOWN; §3.23 pushes KonJND data and gets KonJND UP / AIC-3 DOWN. The two HF
  holdouts pull against each other under *either* supervision, which is a second independent line of
  evidence for the standing limit: **G5 needs an HF feature REPRESENTATION, not more supervision and
  not more data** (consistent with the already-falsified single-MLP agg-head and regime-routed-ensemble
  attempts). Ship implication: the honest champion (no konjnd) stands; konjnd-dense does not earn a
  slot. Note the composite ranks `r5-hon+kon0.5` first (1.404) **only because the konjnd leg is now
  cheat** — a live example of why the honesty-per-(bake,corpus) matrix is load-bearing. Cite:
  `benchmarks/blend_search_r5_2026-07-15.tsv`.

- **3.24 the weak near-lossless (high-tail) rank is NOT the winsor bound — MEASURED, FALSIFIED
  (2026-07-15).** Enabled by the tail stat added the same day (§ the range-restriction fix): the honest
  champion's CID22 **high-tail SROCC 0.434 sits well below its low-tail 0.642** → near-lossless rank is
  the genuinely weak end. The width-10 B9 band could not show this (range-restricted, see
  `[[project-band-srocc-range-restriction]]`). *Thought-why:* "`winsor_pct` fits lo/hi percentiles on the
  TRAINING distribution; bounds too tight clamp near-lossless features to constants (feature-vanishing)
  — exactly the Part-7 mechanism, where recomputing clean p1/p99 bounds lifted B's near-lossless per-img
  SROCC 0.286→0.886 AND CID22 +0.049." *Actual-why:* **flat.** Sweeping `winsor_pct` 0 → 0.02 → 0.05 →
  0.1 → 0.25 → 0.5 → 1.0 (`blend_search.py --round 6`, seeds 1,7,13) moves high-tail only
  0.426/0.438/0.442/0.437/0.433/0.439/0.446 — a **0.020 spread with no monotone trend across a 50×
  change**, and **winsor=0 (no winsorization at all) gives 0.426**, i.e. no better. Low-tail equally flat
  (0.628–0.642). The pre-registered falsification fires.
  **Why this does NOT contradict Part 7:** Part 7's defect was a **fit-corpus/eval-regime MISMATCH** —
  B's *shipped* bounds were fit on a corpus that EXCLUDED near-lossless, so 245/372 features went
  constant there and a recompute recovered them. The honest MLP's bounds are fit on its own training
  data, which already contains near-lossless content (safesyn q100 → butteraugli 0.229), so there is
  nothing for a looser bound to recover. Winsorization is not a near-lossless lever when the fit corpus
  already covers the regime — it is only a lever for repairing a mismatch. (Winsor still slightly helps
  CID22: 0.8862 at the 0.1 default vs 0.8802 at 0 — outlier guarding, not near-lossless.)
  **⚠ SELF-CORRECTION (same day, measured within the hour):** the "next suspects" first written here —
  (a) near-lossless feature-vanishing, (b) training-target saturation, (c) CID22 top-end MOS noise, with
  (b) called "most likely" — were **two wrong inferences stacked on an unverified premise.** Both are now
  MEASURED and (a)+(b) are refuted; (c) is right, and is better described as an intrinsic difficulty floor:
  - **The premise "high-tail = near-lossless" is FALSE.** CID22's high-tail (top 30% by MOS, MCOS
    81.2–91.9) has ssim2 p10 **75.4** / p50 **81.7** / p90 **88.1**, and **0.0000 of it is above ssim2 95**.
    It never enters the near-lossless region at all — it is simply the better half of CID22's ordinary
    compression range, which training covers densely. So (a) cannot apply.
  - **(b) is REFUTED: the target is NOT ceiling-saturated.** safesyn's actual target is `ssim2_gpu`
    (not `human_score`): p50 **68.73**, p90 **91.15**, p99 **95.89**, only **1.86%** above 95 and **0.35%**
    within 2 points of max. That is nothing like the cvvdp saturation §3.20 measured (37% piled in
    [9.8,10]). There is gradient at the top; `ssim2_log_norm` is not indicated here.
  - **(c) is the answer, and it is a DIFFICULTY FLOOR, not our defect.** **ssim2 itself** shows the same
    asymmetry on the same split: ssim2's own low-tail SROCC **+0.649** vs high-tail **+0.463** — versus our
    honest champion's **+0.642 / +0.434**. Our bake tracks an independent, well-established metric to
    within **0.007 (low-tail) / 0.029 (high-tail)**. Ranking stimuli humans rated MCOS 81–92 is
    intrinsically harder than 28–64 (subtler differences, more MOS noise per unit of signal) — the tail
    gap is a property of the task and the corpus, not of the model.
  **Standing lesson:** a low tail/band number is only evidence of a model weakness *after* you check what
  the tail actually contains and what an independent metric scores on the same split. Both checks are
  cheap; skipping them produced a plausible, well-argued, wrong causal story that survived one commit.
  Cite: `benchmarks/blend_search_r6_2026-07-15.tsv`.

---

## 4. Live doc conflicts + inaccuracies to fix (flagged for user, NOT yet edited)

Per the "batch doc corrections, ask before editing" rule, these are surfaced, not changed:

1. **`CLAUDE.md:302-318` "⚠️ V0_8 CID22 SROCC IS INFLATED" box is UNVERIFIED and conflicts with
   the post-revert rules at `CLAUDE.md:2036-2057`.** The box asserts the V0_8 leak (22/49) as
   fact; but it used the same loose d≤16 threshold that §3.2 shows mostly-false-positives, was
   flagged "re-audit at d≤10," and never was. **Proposed fix:** mark the box UNVERIFIED / re-audit
   at d≤10, or run the d≤10 audit to settle it.
2. **`profile_b_methodology_2026-07-12.md §1` names the shipped BHdr as `anchored2` (`373eac56`)**,
   but `profile.rs` + `reproduce_bhdr.sh` ship `7d7f2123` (cvvdp-mix). The methodology §1 table is
   **stale** (its own §3b admits the promotion). **Proposed fix:** update §1 to `7d7f2123`.
3. **The BHdr commit `6b4aae03`, CHANGELOG, and `profile.rs` comment still assert the retracted
   "significant UPIQ win"** (§3.6). **Proposed fix:** annotate with the §7 audit correction (claim
   is non-inferiority, not improvement).

---

## 5. imazen-26 + the diversity gap + the informed retrain plan

**imazen-26** = the imazen-org codec-corpus (`~/work/codec-corpus/imazen-26/`): **2,160 images
across 21 content categories** deliberately spanning the hard cases — real photos (own
phone/camera), Unsplash stock, museum artwork (CC0), **born-digital gov documents (text/tables/
charts)**, **document/manuscript scans incl. bilevel CCITT patents**, **synthetic charts +
line/polygon patterns**, **402 UI screenshots (mobile+web)**, and **910 AI-generated graphics**.
Split = **LSD-origin** (last digit of origin id: {0,2,4,6,8}→train, {1,3,5}→val, {7,9}→test;
`zenmetrics/scripts/picker/origin_split.py` is the one source of truth). dHash-clean vs CID22-49
at d≤10 (min d=12) → adding it does not contaminate the gold holdout. ("26" reads as the 2026
corpus, not a category count — inference, not documented.)

**The gap (why this matters).** The current MLP training set is **photographic-dominant with
synthetic/analytic distortions**: safesyn = synthetic tiles from photo sources; cid22_train =
CID22 photos; KADIS = Pixabay photos with *analytic* distortions. Under-represented: **screen/UI,
line-art/vector/charts, document/scan/bilevel, AI-generated graphics, artwork — and REAL
modern-codec distortions** (the RD the dial must actually rank).

**Best gap-filler:** `canonical-picker-2026-07-01-zensimA` (5 SDR datasets, 5.74M cells) — or
the ready-made LSD-split **`bigcodec_hqdedup_traindigits_2026-07-02.parquet`** (2.32M deduped
rows; verified: `ref_basename` + `human_score`=ssim2/100 ∈[0,1] + 372 `f`-feats, 608+ imazen-26
origins). It fixes both gaps at once (content diversity + real codec distortions). **NOT** KADIS
(more photographic/analytic bias) and **NOT** kadis-hdr (HDR domain — BHdr track).

**Retrain plan that HONORS this history (task #17):**
- **MLP, not linear** — bigcodec poisons a linear CID22 (§2) but MLPs absorb it (§1). This is the
  scientific rationale for doing diversity as the §8.33 MLP. **Still MEASURE** CID22 with it in.
- **Target = ssim2** across all groups (normalize the [0,1] bigcodec/safesyn/cid22 targets ×100 to
  KADIS's ssim2 scale, then standardize) — *not* `score_zensim` (that would distill profile A).
- **Honor the HQ-saturation confound** — ssim2 labels saturate above 0.85 (cvvdp-agreement 0.48);
  densifying HQ-ssim2 *made bakes worse*. Downweight the HQ band or lean on the fill4 cvvdp column
  there; use ssim2 only in its reliable ~30-85 band.
- **Valid selection instruments (§0)** — training-side selection (never peek CID22, never the
  `cid22_train` axis which anti-correlates), full Mohammadi panel + two-panel dial, paired
  significance. Collapse-gate the seed fan.
- **Content-dedup mandatory** — 22.2% of raw canonical rows are byte-identical knob-no-ops (bigcodec
  is already hqdedup, but verify C10 <1%).

---

## 6. Standing guards this history produced (the scar tissue)

- **Two-panel eval mandatory** (rank + dial; G1 dynamic-range + monotonicity) — from §3.4/§3.5.
- **Full Mohammadi panel, SROCC-only BANNED** (SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE) — from §3.3.
- **Paired significance + Westfall-Young family correction; a holdout is never a selection axis** — §3.6/§3.12.
- **Pre-registered gates + one holdout look** — from the post-selection incidents.
- **`_validate_metric_columns()`, grid quarantine, fail-loud joins, assert-features-non-constant** — §3.1/§3.4/§3.15.
- **Regime tags (`--features-name`), runtime-parity spline caps, `[training].trainer_commit` pin** — §3.9/§3.11/§3.13.
- **`|SROCC|` for konjnd; `pjnd_target` not `human_score`** — §3.16.
- **CID22-49 / AIC / UPIQ / SDR25 are holdout-only; UPIQ-380 is BURNED for the BHdr linear family.**
- **UPIQ-panel every BHdr candidate before ship** — §3.13 (hdranch3).
- **Keep bigcodec/imazen-26 OUT of *linear* mixes; MLPs only.** — §1/§2.
- **Non-photo standing gate** (added 2026-07-15): `bake_verdict` scores a held-out imazen-26
  diverse-content axis (`nonphoto` corpus) by default + a **G-NP gate** (SROCC <0.88 = content-weak,
  <0.50 = crash). Every photographic-only bake (B/A/§8.33) flags content-weak (~0.86) — the
  blindness the 6 photographic corpora can't see. Dashboard: `scripts/v_next/bake_dashboard.py`.
- **NO GRACEFUL SKIPS in eval** — a missing corpus fails loud with an R2 fetch hint; all 7 eval
  corpora are mirrored to `s3://zentrain/eval-corpora/` (the exact `2026-05-15-full-features`
  files `bake_verdict` reads). Never silently drop an axis.
- **Winsorize before the scaler on any bigcodec/imazen-26 train** (combined `[p0.1,p99.9]`,
  baked as `WinsorP99` transforms) — de-poisons the IW-block extraction garbage (§3.4-family). §8.35.
- **Never MSE-regress a log-expanded target** (e.g. `cvvdp_log_norm`) — use raw or a rank loss. §3.17.
- **Run `check_table_integrity.py` on any mix before training it** (added 2026-08-04) — the
  structural gate the project went two years without. Covers target finiteness/range/degeneracy,
  feature finiteness + constant columns + unguarded tails, duplicate rows, teacher-twin row
  correspondence + agreement, and eval leakage by reference identity. `--mix-from-spec <bake>.
  bin.spec.json` audits a whole recipe from its embedded repro. §3.18.

### §3.18 — the SOTA-944 mix data-integrity audit (2026-08-04)

Report: `benchmarks/data_integrity_audit_2026-08-04.md`. Plan:
`benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX G.

**Thought-why:** after the KADID inversion (Appendix F) — found by the project's FIRST
orientation gate, on its first run, six weeks late — the open question was whether the rest of
the mix hid similar defects, and whether the mix had ever been chosen on evidence.

**Actual-why / what was found:** the mix is clean where it would have been catastrophic and
dirty where nobody was looking.

- **Eval leakage is ZERO** — 10 training legs × 10 eval corpora, every pair empty. CID22
  explicitly: `cid22_train`'s 201 references ∩ the `cid22val` 49-reference holdout = ∅, and
  `cid22_train`'s target is `ssim2_gpu`, never human MOS. The VALIDATION-ONLY rule holds at both
  the reference and the target level.
- **Content-dedup WAS applied** — max duplicate-row mass 3.66% (bigcodec), far below the 22.2%
  documented pre-dedup rate. The mandate in the content-dedup section is being honored.
- **`tkadis` is a training-signal conflict** — the kadis teacher twin ranks its own rows at
  signed SROCC **+0.2485** vs the base kadis target, with a systematic +0.579 median offset, at
  **7.87% of sampling mass against the base leg's 2.36%**. The clip/affine explanation is
  FALSIFIED (0.05% clipped; SROCC identical on the unclipped subset; SROCC is affine-invariant).
  It is a genuine teacher generalization failure on the KADIS distortion distribution.
- **Effective sampling mass is ~uncorrelated with row count** and had never been written down.
  The trainer picks a group by `train_w / Σ train_w` then draws rows uniformly *within* it, so
  pair share is **independent of row count**: `konjnd_bpg` is 1.03% of rows → **18.90%** of
  pairs; `bigcodec` is 26.71% of rows → 7.87%. **The weights ARE the mix.** Table:
  `benchmarks/data_integrity_sampling_mass_2026-08-04.tsv`.
- **9 of 11 legs cannot be orientation-checked against humans at all** — they carry metric
  (ssim2) or teacher targets. KADID's inversion was catchable *because KADID is one of only two
  legs where an external check was ever possible*. `check_target_orientation.py --provenance`
  now reports this explicitly so the gap cannot be mistaken for a clean bill of health.
- **The canonical promotion drops the quality key**, so ladder monotonicity is unauditable for
  safesyn / cid22_train / konjnd_bpg (17.5% of rows). Where the key survived, monotonicity
  PASSES cleanly (bigcodec: 9,228 ladders, median SROCC(q, target) +0.963; kadis: per-source
  median −1.000 against severity). **Carry the quality key into every canonical table.**
- **The 39 never-populated feature slots are an EXTRACTOR property, not a data gap** — zero
  slots are constant in only *some* groups. Independently reproduces `bake_contrib`'s count and
  classifies it. Prune candidates only (the `n_inputs()` vs `caller_input_width()` hazard, E.9).

**Two audit-side errors caught before publication, recorded so they are not repeated:**
`pair_tie_prob ≠ row_tie_rate` (KADID reads 99.60% by the wrong one, 0.876% by the right one —
the trainer drops *pairs*), and `signed_cbrt` is a tail guard just like `winsor_p99` (f38's 776×
excursion is guarded, not unguarded). Both are now encoded in the gate.

**Registered non-conclusion:** no check disqualified the mix, therefore **the mix-composition
question is UNANSWERED**. Absence of defects is not evidence of optimality; establishing an
optimum needs a weight sweep with held-out scoring, which this audit did not run. Do not cite
this audit as evidence that the mix is good — only that it is not broken.

---

*Maintenance: update this doc whenever a verdict is added, re-framed, or a confound is found.
Prefer editing here in-place over a handoff note. Every claim keeps its citation.*

### §3.24 — imazen-26 root-source correction: wrong estate copy, provenance-derived sharing, manifest split-column fix (2026-08-27)

**Thought-why:** `/mnt/v/imazen-26` was "the imazen-26 SDR estate", so the
2026-08-27 dHash audit indexed it as its refs; `imazen26_manifest.tsv`'s
`split` column was assumed meaningful; content sharing between synthetic-v2
and the eval slices was assumed to need dHash to find.

**Actual-why:** that directory is the INSPIRATION/collection copy (the wsl box
had already quarantined its copy as `imazen-26-inspo`; this box's rename had
never happened). The canonical estate is the **`imazen/imazen-26` git repo**
(CORPUS-MANIFEST membership oracle + canonical split manifests + sha-pinned
variant-set registry) with official png-v3 derivatives; eval/picker origin
`o_NNNN` IS the canonical 4-digit id. The manifest TSV's `split` column was
stale bookkeeping (wrong on 1,239/2,157 rows; header mislabeled `sha256`) —
corrected 2026-08-27 to the canonical last-digit rule (consumers verified;
pre-fix bytes in `.pre-splitfix.bak`). And sharing is **provenance-derivable**:
generator tokens live in the filenames on both sides, and cross-id duplicate
families follow the name grammar — dHash is only the verifier (it also
false-positives on flat/line content and misses non-picker-covered ids).

**Outcome:** chain byte-verified (repo → registered picker set → local
mirrors); sharing = 68 exact-token ids + 166 split-piercing family ids;
realized eval inflation MEASURED ≈0 at certain tier AND the 25–32% upper bound
(median Δ +0.0043, max 0.0143, nonphoto positive). Annotation
`imazen26-nonphoto-sharing-provenance-2026-08-27` in force; exclusion = board
D1. Records: `benchmarks/imazen26_dhash_audit_2026-08-27.md`, imazen-26
`benchmarks/split_crossid_dupes_2026-08-27.md` + `derive_sharing_provenance.py`.

**Rules:** join imazen-26 data by the 4-digit id, never by dHash; derive
sharing/duplicates from provenance first, verify with dHash second; never
index `/mnt/v/imazen-26*`; a by-design content family crosses the split iff
its id offsets flip parity (family-aware split is the structural fix).

### §3.25 — the family-aware purity program EXECUTED (2026-08-28, user decisions)

User calls (recorded in the audit md ★REGISTERED section): structural
family-aware re-slice + full-board rescore (done — 280 rows grafted, 11
ensembles + 2 wrong-regime rows annotated); measure hfnlproxy/instruments
(done — SDR instrument clean at d≤2; HDR instrument: hdr_v3mix carries 7/9
census scenes — judge/training overlap annotated, freeze gates unaffected);
training policy = **purge + family-aware**: `benchmarks/
synthv2_channelA_purge_2026-08-28.tsv` lists the synthetic-v2 files that
leave the metric training lineage for all FUTURE trainings, and future
picker/bigcodec/HDR train views bucket by `split_map_family.tsv` + hold the
HDR instrument scenes out entirely. Existing bakes stand, era-tagged.
D3 one-shot wirings executed (zenjpeg/jxl/svt). D4 decided: hdrgrid stays
era-B. **Rules for every future corpus build:** consume the family split,
apply the purge list, and keep instrument scenes out of training views.

### §3.26 — v1's 372-feature vector width: RESOLVED 2026-08-30 (it was SIZE, not batch)

**STATUS: RESOLVED.** Root-caused, fixed in the owner, and gated —
commit `f9fac41e` (`fix(v1-372 width): reflect-pad at EVERY pyramid entry`) on
`main@origin`; gate `zensim/tests/v1_feature_width_pure_function.rs`; full
record with every measurement `benchmarks/v1_width_defect_2026-08-30.md`.
**The BATCH framing below is retracted in turn** — this entry keeps both
retractions because two different wrong explanations were registered here and
the second one is the more misleading of the two.

**What was seen.** Re-extracting the R1b eval slices at the v1-372 regime
produced RAGGED CSVs: 453 of 6,953 imazen26 rows, 422 of 6,142 nonphoto, 493 of
7,717 hfnlproxy (~6.5 % each) carried **279** feature columns instead of 372.
279 = 3 scales × 3 channels × 31; 372 = 4 scales × 3 × 31 — one whole scale
missing. The rows are not empty: a short row carries ~268 non-zero values, i.e.
a real 3-scale computation.

**Retraction 1 (the original entry).** "v1's vector length is size-dependent —
a rendition too small for the 4th scale emits 3 scales" was rejected on
2026-08-30 because `512x384` appeared among the short rows, the same size
appeared in both sets, and 259 of 957 references carried BOTH widths.

**Retraction 2 (what replaced it, also wrong).** "The width is a function of
the BATCH, not of the pair" — evidenced by the same 453 pairs re-run as their
own batch giving only 33 short and 5 run alone giving 0. **Both numbers fail to
reproduce.** A binary built from the pre-fix tree (`6d0a393a`) on the exact
pair lists gives **5 short of 5 alone, 453 short of 453 alone, 453 short of the
6,953-row batch** — the width never moves, and the row VALUES are byte-identical
across all three compositions (pre-fix and post-fix alike). The likeliest source
of the 33/0 reading is a re-run routed through a `Zensim::compute*` entry (which
reflect-pads, so 0 short) and/or a pair list rebuilt from `ref_basename`, which
§8.5(a) of the R1b doc itself records as **not row-unique**.

**The truth: it is a pure function of `(W, H)`.** The scale walk starts at
`w = simd_padded_width(width)` but plain `h = height` and stops at `w < 8 ||
h < 8`, so a 4-scale pyramid needs **`simd_padded_width(W) ≥ 64 AND H ≥ 64`**
(i.e. `W ≥ 49 && H ≥ 64` below 497 px). The predicate `2 + n_scales(W,H)·3·31`
reproduces the field count of **all 20,812 stored rows with ZERO errors**; every
short row's min side is in `[36, 55]`; the short and full size sets are
**disjoint**. The width asymmetry is why "too small" looked falsified: `54x96` is
FULL (54 → 64 by SIMD alignment) while `96x54` is SHORT, and `62x96` is FULL
while `48x64` is SHORT. The size figures quoted above do not describe the short
rows under any parse: across all three slices they span **13 distinct `(W, H)`
classes**, min side `{36…55}` and max side `{64, 96}` — not 168 sizes spanning
36…1024 — and `512x384` is **not** among them.

**Mechanism.** `compute_with_config_inner` (`metric.rs:3145`, behind every
`Zensim::compute*`) reflect-pads any sub-64 side before the walk. Three entries
did not: `compute_zensim_with_config` (`metric.rs:4800`, `training`) returned a
**silent short vector** (93/186/279 wide, no error) — and **both** v1-372
extractors call it (`extract_features_372col.rs:195`,
`v2_ab_extract.rs:319`), which is exactly why the identical short set came from
both tools and both flows; `compute_zensim_with_ref_and_config`
(`metric.rs:706`, `training`) and `Zensim::compute_with_ref_into`
(`metric.rs:2271`, a **product** API) **panicked** `scale 0 width mismatch`.

**Fix.** One owner for the decision — `metric::needs_pyramid_pad(w, h,
num_scales)` + `min_pyramid_dim_for_scales` + `reflect_pad_for_scales` — used at
all seven pyramid entries. `MIN_PYRAMID_DIM` stays 64; the threshold is now
`num_scales`-aware so `--num-scales 5/6` cannot truncate either.

**Consequence for existing data — MEASURED, and it is benign.** Every canonical
372 parquet is full width with real values; nothing is NaN- or zero-padded:

| table | rows | last-93 block all-zero | non-finite |
|---|---|---|---|
| `cid22_features_372col_2026-05-15` | 4,292 (complete corpus) | 0 | 0 |
| `kadid_features_372col_2026-05-15` | 10,125 (complete corpus) | 0 | 0 |
| `konjnd_features_372col_2026-05-15` | 1,008 (complete corpus) | 0 | 0 |
| `imazen26_test_120k_2026-07-16` | 7,844 | 0 | 0 |
| `nonphoto_features_372col_2026-07-15` | 8,241 | 8 (0.10 %) | 0 |

A ragged CSV cannot become a parquet silently — the builder raises
(`ArrowInvalid: Column ... expected length 6953 but got length 6500`) — so a
short row is always either a build failure or an explicit drop, never a padded
row. The 8 nonphoto all-zero rows are the identical-input signature, not
truncation. **Confirmed 2026-08-30 by direct measurement:** a header-level
dimension scan of all **149,195** pairs across the 11 canonical legs finds
**0 pairs** that could have truncated; re-extracting cid22val (250), kon504
(504) and safesyn (250) gives 372 on every row, alone == batch byte-identical,
and the **pre-fix and post-fix binaries agree BYTE-FOR-BYTE** on all 1,004 rows.
(Unrelated and pre-existing: a fresh extraction differs from the *stored*
2026-05-15 / 2026-08-29 tables on most slots — that is the §8.5(b) extractor
drift, not this defect; the two binaries agreeing byte-for-byte is the proof.)

**The 944 regime does not have this problem.** The folded/append path emits a
fixed width, so R1b's 944 tables are ragged-free by construction. Measured:
`foldapp2pools` on the 453 sub-64 pairs emits 946 columns on every row, and the
pre-fix and post-fix CSVs are BYTE-IDENTICAL.

**Where it did land, still open.** The three R1b eval slices are the only place
sub-64 renditions met the defective extractor. `build_r1b_samepair_roots.py`
dropped the 1,368 short rows (counted, never silent), so
`r1b-samepair372-2026-08-30` is a size-correlated 6.5 % restriction of its
population, and `r1b-372root-2026-08-30/` carries **three dangling symlinks**
into `r1b-372slices-2026-08-30/`, which only ever received
`ext_konjnd_jpeg_val.parquet`. Full-width replacements now exist as CSVs at
`/mnt/v/output/zensim/v1width-fix-recheck-2026-08-30/` (sha-manifested,
`build_commit` stamped, with the exact affected-pair lists); promoting them to
parquet and re-cutting the same-pair roots at 6,953 / 6,142 / 7,717 rows belongs
to the keyed-rebuild lane and is REGISTERED, not executed.

**Guard.** `v2_ab_extract` refuses to write a table whose row count differs from
the pairs count and exits 3 (`ZENSIM_AB_ALLOW_MISSING=1` is the caller's visible
opt-out). That guard catches a DROPPED row; the new width gate catches a SHORT
one. Record: `benchmarks/r1b_keyed_rebuild_2026-08-30.md` §8.5(d) +
`benchmarks/v1_width_defect_2026-08-30.md`.

### §3.27 — the STORED 372-col masked/IW block was a function of the thread count (2026-08-30)

**Thought-why:** "the 2026-05-15 canonical 372 tables and today's extractor
disagree on masked/IW — three and a half months of extractor evolution."
(§8.5(b) of `benchmarks/r1b_keyed_rebuild_2026-08-30.md`, priced there at
+0.0060 SROCC for B on CID22 and left as an unexplained cross-era gap.)

**Actual why:** ONE commit, and the stored side of the comparison was never
reproducible in the first place. `2dab8f30` (2026-05-17,
*"principled per-channel H-blur activity for masked/IW features"*) replaced the
activity-map reference — which read `bufs.mu1` at strip-**overlap** rows that
the fused V-blur never writes, i.e. whatever the buffer-reuse cascade left there
— with a per-channel `H_blur(src_c)`. Its own message names the blast radius:
*"Affects masked (228..300) and IW (300..372) feature blocks. Basic 228 features
are unchanged."* Combined with `6af83b60`'s pre-fix band layout
(`num_bands = rayon::current_num_threads().min(total_strips)`, made geometry-only
2026-06-09), the thread count chose where those overlap rows fell — so the
pre-fix masked/IW block was **machine-dependent**.

**Measured** (`benchmarks/v1_extractor_drift_2026-08-30.md`; artifacts +
`_MANIFEST.json` at `/mnt/v/output/zensim/v1-extractor-drift-2026-08-30/`):

- **The stored table does not reproduce at its own build commit.** A probe built
  at `58e6f8d8` — the commit `_MANIFEST.md` records for the 2026-05-15 tables —
  run at `RAYON_NUM_THREADS` 1 / 2 / 8 / 28 produces **four different**
  504×372 outputs. T=1 vs T=28 moves **100 % of rows on all 144 masked/IW slots,
  up to |Δ| 0.086**, while basic + peaks have **zero** cells outside the golden
  tolerance. At HEAD the same four runs give **one** md5.
- **Blast radius is exactly masked+IW.** stored-vs-HEAD on cid22val (4,292
  pairs): `f0..155` and `f156..227` **bit-identical** (max_abs 0), `f228..371`
  differ on 100 % of rows (max_abs 0.0374 masked / 0.1235 IW). Pixels are
  therefore identical, and HEAD's with-ref path == HEAD's plain path
  bit-for-bit, so neither decode nor entry path is a confound.
- **Nothing since.** `2dab8f30` → HEAD is **0 cells over tolerance** on
  4,292 × 372 (residual 5.55e-17 on 18 scale-0 IW slots).
- **Era map:** `2026-05-15-full-features` is PRE-fix.
  `ext720-canonical-2026-07-22` and HEAD are the SAME post-fix era (they differ
  only on the 439 rows where the zen_io decoder disagrees with the `image`
  crate, and those differ in basic too). `r1b-pools944-2026-08-30` is post-fix
  but on the FOLD path (the documented `folded720_*` padded-width class).
- **`canonical-2026-05-21/train/{kadid,tid}.parquet` carry the SAME pre-fix
  values** — row-order identical to the 2026-05-15 root on `f0`, `f228`, `f300`,
  `f353`. So the whole training lineage of the 372 era inherits them.

**Product consequence (the reason this is a §3 entry and not a footnote):**
shipped **Profile B** has 23 of its 95 live inputs in `f228..371` and its single
largest-magnitude input is `f353 = iw/s2/c2/iw_mse` (L2 norm 182.4, 2× the next).
Same bake, same pairs, same pixels, matched row sets, two feature tables:
CID22 SROCC **0.87638 → 0.88212**, KonJND **0.54665 → 0.64967**, AIC-3
0.77743 → 0.79410, TID 0.78866 → 0.79691, KADID 0.82008 → **0.80426** (KADID is
B's train==val CHEAT corpus, so a memorization score falling is the expected
sign). Per-pair the **dial** moves by mean **−4.98** (CID22) / **−5.86**
(KonJND) zensim points, 99.9 % / 100 % of pairs by more than 0.5 points, max
17.4. The product API (`Zensim::compute` at `codec_target`) matches the
fresh-root prediction to 8 decimals on 10/10 sampled pairs — so **the runtime B
is not the evaluated B**, and the published B numbers are stored-root values.

**Why nobody caught it in May.** The 2026-05-20 canonical-build audit
(`~/work/zen/_ml-inventory-2026-05-20/10-canonical-build-audit.md`) that
`zensim/CLAUDE.md` cites as "bit-equivalent … no build drift; trustworthy as-is"
**sampled only `f0..f99`** — its own §1 says *"emits f0..f99"* and its tolerance
is `max_abs_diff(extracted_f0..f99, parquet_f0..f99)`. `f0..f99` is entirely
inside the basic block, the one block that did NOT drift. It ran at `fdd1b8f6`
(2026-05-19), already past `2dab8f30`, so a single masked slot in the sample
would have caught this three months earlier. Its §5 softening of the
`DATA_PROVENANCE.md` "semantically incompatible" warning rests on the same
100-column sample and does not extend to `f156..371`.

**Guard.** `zensim/tests/v1_feature_width_pure_function.rs` gained
`v1_372_is_bit_identical_across_rayon_pool_sizes` and
`v1_masked_and_iw_blocks_are_thread_invariant` — rayon pools of 1/2/3/5/8 (and
1/2/4/7/16) on both `compute_zensim_with_config` and `Zensim::compute`, at sizes
spanning several `STRIP_INNER = 32` strips. The extractor was NOT changed and no
tolerance was widened: the pre-fix values were undefined buffer contents, not a
different-but-valid definition.

**Registered, not executed:** rebuild the canonical 372 root at HEAD as a NEW
dated root (cheap — cid22 8.8 s, kadid 14.9 s, tid 4.2 s, konjnd 2.9 s, aic3
14.2 s at 8 jobs) rather than overwriting in place; re-verdict every
372-input `uses_f156_371` board cell; re-extract B's training legs (safesyn
196,086 + cid22_train 17,611 + kadid 10,125 + tid 3,000 + `hdr_v3mix`) and
consider a retrain — a fleet wave, not a step. `aic4` cannot be refreshed: its
source CSV under `/mnt/v/backups/...` no longer exists on this box.

**Separate defect found in passing, NOT fixed:** `zensim-validate --extract-only
--format tid2013` yields **2,880 of 3,000** TID pairs today — 120 rows silently
dropped on decode/extract failure, surfaced only as a `2880 valid pairs` count.
Same "silent skip" class already documented for `dataset_metric_baseline`.

### §3.28 — the DATED current-extractor 372 eval root, and the era shift priced per model (2026-08-30)

**Thought-why:** "§3.27 declared the stored 372 tables stale; rebuild the root
and re-verdict the lineage." (Registered follow-ups (a) + (b) of
`benchmarks/v1_extractor_drift_2026-08-30.md` §4c.1–2.)

**Actual why + what shipped:** a NEW dated root at
`/mnt/v/zen/zensim-training/2026-08-30-full-features-372/` (`_MANIFEST.json`,
`build_commit ea16c7ee`, per-file sha256 + row accounting + per-corpus ERA +
per-slot drift-vs-stored), never an in-place overwrite — the 2026-05-15 root is
untouched and every published 372-regime number keeps its substrate. Eight of
the fourteen default `bake_verdict` corpora were re-extracted through the SAME
tool the stored table used; six (aic4, nonphoto, imazen26, sdr25, hfnlproxy,
hf_nearlossless) are byte-copies because their distorted material is
bigcodec/R2 encodes or, for aic4, a source CSV that no longer exists — so an
era delta of zero on those is a **structural identity, not evidence**. Plus a
current-era `kon504` ruler + a `kon504/` one-file side root.

**Measured (`benchmarks/eval372_current_root_2026-08-30.md`):**

- **The era shift is MODEL-SPECIFIC, not a constant** — 11 bakes × 2 roots ×
  the default 14-corpus list, same instrument: from **exactly 0.00000** (three
  basic-block-only bakes, all 15 corpora) to **|Δ| 0.489** (`cl_tfm_LQ_MLP`,
  KonJND). No correction factor exists for a published 372-era number.
- **41 ordering flips.** Shipped **B goes 4th → 1st on CID22** among
  {B, `blend_2L_H128`, `cl_tfm_LQ_MLP`, `Ebothg_scr05`}; `cl_tfm_LQ_MLP` goes
  **1st → last** on KonJND (0.761 → 0.272) and on AIC-3; the composite leader
  changes `cl_tfm_LQ_MLP` → `blend_2L_H128`.
- **The 2-layer blend's headline CID22 win over B (+0.004) is an era artifact** —
  on the current extractor it is **−0.0002**, i.e. B is ahead. Its TID +0.062 and
  nonphoto +0.088 survive; its KonJND deficit deepens (−0.038 → −0.145).
- **csiq / live / pipal (built 2026-07-18) are BIT-IDENTICAL to a HEAD
  re-extraction** — 0 slots differing, max_abs exactly 0 on 866 / 779 / 21,800
  rows. §3.27's commit-level era map is now a direct measurement on three
  corpora, and it survives six weeks of extractor work including the blur
  rewrite. Same check on cid22 + konjnd vs the drift lane's own `f9fac41e`
  extraction: 0 cells over tolerance — the `714da506` / `8a98a286` BIT-EXACT
  claims hold on corpus data.
- **The 372 dial + corruption grids are themselves pre-fix-era** (extracted
  2026-05-29 / 05-28, i.e. after `2dab8f30` but before `6af83b60`) and are NOT
  re-extractable: their pairs TSV names the `q<X>.png` decode cache deleted
  2026-06-22 (**2,560 of 2,560 dist paths missing**). Both eras of a
  stored-vs-current re-verdict read the same grid file, so the identical dial
  panel is an identity, not a clean bill of health.
- **kon504 is two files under nearly one name.** R1b's keyed 504 rebuild is
  post-fix (HEAD reproduces it bit-for-bit — W-LIN r7, `df931814`); the 372
  root's `konjnd_jpeg504_372_2026-08-29.parquet` is a byte-exact subset of the
  **pre-fix** konjnd table and carries the full §3.27 signature (masked 34,525 /
  IW 35,254 cells over tolerance on 100 % of rows). Both lanes are right about
  different files.

**CORRECTION to §3.27's source doc (§3b of `v1_extractor_drift_2026-08-30.md`):**
its KADID / TID / AIC-3 rows are invalid — the study's `mkroots.py` aligned
stored↔fresh on `(ref_basename, human_score)`, a key that is **not unique** on
those corpora (stored rows in repeated groups: KADID 64.8 %, AIC-3 100 %, TID
24.2 %), so a whole group collapsed onto one fresh row. Its own `freshroot/`
tables carry the evidence: **aic3 100 distinct rows of 600**, kadid 6,227 of
10,125, tid 2,505 of 2,880; **cid22 and konjnd 0 duplicated**, which is why the
headline CID22/KonJND numbers stand. Positionally aligned: KADID 0.82008 →
**0.80847**, TID **0.78683 → 0.77852**, AIC-3 0.77743 → **0.76501** — the TID
and AIC-3 deltas **change sign**, so "on every genuine holdout the runtime B is
better" is falsified (AIC-3 goes DOWN). Mechanism and decision unchanged.

**Loader fix that came with it (`2d94890c`):** `load_tid2013` forced the
reference stem upper-case while TID2013 ships `i25.bmp` LOWERCASE, so 120 of
3,000 rows named a nonexistent path and were dropped with the loss visible only
as a row count (§3.27's "separate defect found in passing"). Both sides now
resolve through a case-insensitive index and an unresolved label row is FATAL
unless the caller sets `ZENSIM_ALLOW_MISSING_PAIRS=1`. The new root's TID is
**3,000/3,000**, and the recovered 120 rows are bit-identical to the stored
table in basic+peaks — they were absent, not different.

**Registered, NOT executed:** the board is NOT regenerated (ready-to-promote
verdicts at `/mnt/v/output/zensim/eval372-roster-2026-08-30/json/<label>_new.json`);
the dial/corruption grid rebuild needs a decode pass first; aic4 stays pre-fix;
B's training legs stay pre-fix (the retrain is still a fleet wave); BHdr's own
PU-linear HDR route is still unmeasured — the `BHdr_sdr_route` row bounds only
the SDR-route sensitivity (≈0.002 SROCC).

**Registry:** `benchmarks/eval_annotations.json` gains
`eval372-stored-root-thread-dependent-2026-08-30` (invalidated),
`eval372-basic-only-bakes-era-independent-2026-08-30` (annotated — measured
Δ 0.00000, the invalidation does NOT apply) and
`dial372-grid-thread-dependent-era-2026-08-30` (annotated).

**SAME-DAY FOLLOW-UPS (2026-08-30).**
1. **The default `--features-root` WAS flipped** — user directive, executed by
   the board lane in `a25d1b80`: `--regime 372` now defaults to the
   current-extractor root through the single constant
   `zensim_validate::eval_roots::DEFAULT_FEATURES_ROOT_372`, and every verdict
   prints its ruler. A **stored**-era read now needs the explicit flag
   (`STORED_FEATURES_ROOT_2026_05_15`) — the reverse of the discipline in force
   when the roster above was run.
2. **Board-row attribution CORRECTED: 7 of the 9 rows were never stored-root
   reads.** They are `regime:"720"` ext720 reads
   (`board372-row-read-on-ext720-root-2026-08-30`), which this lane reproduced
   independently: board `cl_tfm_corruption_LQ_MLP_s13` vs a fresh
   `bake_verdict --regime 720 --corpora cid22` is **BIT-EXACT on 4,292 pairs**
   and **96.4 points** from a stored-372 read; board **B** and
   **`T_appT_b372_lam1e-3`** are **BIT-EXACT against this lane's stored-root
   re-verdicts** (and 17.4 / 6.09 from the current-root ones), so they are the
   only genuine stored-era board rows and **B's pair is the board's only clean
   era A/B**. The three basic-only rows differ from the stored and current 372
   runs by the SAME amount (0.294/0.294, 0.797/0.797, 0.502/0.502) — an
   era-independent offset, i.e. the folded-720 space, not an era. The two
   affected registry scopes were NARROWED in place to the 2 genuine rows (the
   `kadid-ext-root-inverted` precedent), so no post-fix row carries a false
   era-stale badge. **§3.28's roster science is untouched — it never used a
   board row**; every number came from this lane's own paired runs on the two
   roots.
3. **The provenance gap that forced that archaeology is CLOSED**:
   `bake_verdict --full-json` now records a `features_root` block (path,
   registered era label, root `_MANIFEST.json` sha256 + declared regime, and the
   per-corpus file sha256s it actually read), so "which ruler produced this row?"
   is answerable from the artifact instead of by re-running and diffing
   predictions.

### §3.29 — the default eval root IS the current-extractor root, and what the board rows turned out to be (2026-08-30)

**Thought-why:** "§3.28 left two follow-ups open for governance: flip
`bake_verdict`'s default `--features-root`, and put the current-era verdicts on
the board." (User decision; both executed the same day.)

**Actual-why / what shipped.**

1. **Default flipped, and it now has ONE owner.** The 372 root path was a string
   literal in ten `.rs` files. It is now
   `zensim_validate::eval_roots::DEFAULT_FEATURES_ROOT_372` =
   `/mnt/v/zen/zensim-training/2026-08-30-full-features-372`, with
   `STORED_FEATURES_ROOT_2026_05_15` naming the previous default (which the probe
   and trainer bins now reference BY NAME, so their era choice is deliberate and
   visible rather than accidental). `bake_verdict` and `bake_compare` read the new
   constant; `bake_dial_refit gate`'s default corpus moved to the same-named file
   under the new root. **Nothing was rewritten and nothing was deleted** — the
   2026-05-15 root stays on disk and stays a valid STORED-ERA read; the flip only
   changes what a *flagless* invocation means going forward.
2. **Every verdict is now self-describing.** `bake_verdict` prints
   `bake_verdict: features-root era — <label> :: <path>` before it loads a corpus
   (`eval_roots::era_of`, which labels the four registered roots and reports an
   unregistered one as UNKNOWN rather than guessing). Two tests pin the default
   (`default_features_root_is_the_current_extractor_372_root`,
   `explicit_features_root_overrides_the_default_and_relabels_the_era`) plus two
   in `eval_roots`.
3. **VERIFIED:** a flagless `bake_verdict --bake <shipped B>` and the same run
   with `--features-root <new root>` produce a **byte-identical** `--full-json`
   (sha256 `9596f1bd9f3b2166612866f830e36079675f470870162b73a9ecd2c6d756b2c7`;
   the markdown differs only in the wall-time line), and the flagless numbers
   reproduce §3.28's current-era column exactly (CID22 0.8821166166, KonJND
   0.6496694639, KADID 0.8084738650, TID 0.7785195153, AIC-3 0.7650123966,
   composite 0.8407364995733521).
4. **NOT in the flip:** the 372 dial + corruption grids. They are their own
   pre-fix files (2026-05-29 / 2026-05-28), not re-extractable without a decode
   pass, and stay annotated `dial372-grid-thread-dependent-era-2026-08-30`.
   Training legs are untouched and still pre-fix.

**The board, and a MEASURED surprise.** The 11 current-era verdicts are promoted
as `<stored name>@cur372` rows (`benchmarks/board_era_rows_2026-08-30.md`), the
stored rows kept and gate-verified byte-identical. Pairing them exposed that
**7 of the 9 "stored-era" 372-class board rows were never reads of the 2026-05-15
root**: each is stamped `regime: "720"`, and a fresh `bake_verdict --regime 720
--corpora cid22` reproduces its per-pair predictions bit-exactly, i.e. they came
from `ext720-canonical-2026-07-22`, whose masked/IW block is POST-FIX. Those rows
already agree with the current-extractor read to ≤2e-4 on CID22 while differing
from the true stored-root read by up to 0.0153 SROCC. Only
`b_sdr_linear_cid80_inclwinsor_dense_dial` and `T_appT_b372_lam1e-3` (stamped
`regime: "372"`) are genuine stored-root reads. §3.28's era table is unaffected —
it compared `_old` vs `_new` from one instrument and never used a board row.

**Registry:** `benchmarks/eval_annotations.json` gains
`board372-row-read-on-ext720-root-2026-08-30` (annotated, 7 cells — the scope
correction above), `eval372-current-root-copied-corpora-2026-08-30` (annotated,
11 cells — six of the new root's fourteen corpora are byte-copies, and 39.5 % of
`product_composite`'s weight rides on them) and
`dial372-grid-thread-dependent-era-current-rows-2026-08-30` (annotated, 11 cells).

### §3.30 — ERA-3: v1 stopped pooling phantom columns (option C, 2026-08-30)

**Thought-why:** the fold's `f0..371` disagreed with the buffered v1 path, and
the working assumption (twice) was that the FOLD was the odd one out — first
that it was a tolerance question, then that the fix was to make the fold
reproduce v1 by pre-padding its input.

**Actual-why.** Measured, it is the reverse. **Buffered v1 pooled columns that
are not in the image.** `simd_padded_width` rounded the width up to a multiple
of 16 and added a *further* 16 whenever that landed on an even multiple of 16
at or above 512 (an L1d set-aliasing dodge); `mirror_pad_columns` filled the
extras by reflect-101; and the scale walk then pooled them. The fold never
padded. So the fold was already computing the correct statistic and v1 was not.

Consequences, all measured:

* Divergence up to **81.6 % relative** on a pool slot (200×150) and 17.4 %
  (127×93) — not a tolerance question.
* **Every common production width was in the divergent class**: 512→528,
  576→592, 768→784, 1024→1040, 1152→1168, 2304→2320. Tight widths (multiples
  of 16 below 512, and the non-bumped alignments above) were bit-identical
  across eras and do **not** change era. It is a per-ROW property of the
  width, not a per-table one.
* The h=93 "residual" reported mid-investigation was an artifact of the
  rejected pre-pad workaround, not of either walk. **Under C it does not
  exist.**

**What shipped (era-3, `56bbcda2`).** `blur::pyramid_plane_stride` returns the
width and is the single greppable owner; `mirror_pad_columns` and its three
call sites are deleted (with no padding they could never fire). The fold
needed no change, and that is verified structurally rather than numerically:
its production path never references the owner — every occurrence in
`feature_v2.rs` is a doc comment or test code — so **the 944 regimes are
unchanged by construction and no 944 table or model is invalidated.**

**It is also cheaper.** Buffered v1-372 instruction counts: **−9.02 % at 576,
−7.37 % at 1152, +0.00 % at the tight-width control 592**. The aliasing
rationale in the old doc is stale — on the current kernels the bump cost 7-9 %
at exactly the sizes it applied to. That is why C ships as "no padding" rather
than "padded buffers with pool-width exclusion": the latter would still blur
the phantom columns and is strictly slower.

**The golden gate was structurally blind, and that is the headline.**
`GOLDEN_SYNTHETIC` (64×64) and `GOLDEN_REAL` (96×96) are both stride-invariant
widths, so the golden set sat entirely in the tight class — it would have
passed however wrong the padded class was. **Their values are UNCHANGED by
era-3; they were re-verified, not re-pinned**, and that they still pass is the
evidence. A third fixture `GOLDEN_NONTIGHT` (200×150, procedural) now covers
the class. Negative control run: reintroducing era-2 padding fails **only**
the new fixture — synthetic and real both still pass.

**Gate meanings that inverted** (sanctioned re-pins, each a tightening — no
tolerance was widened anywhere):

| gate | era-2 meaning | era-3 meaning |
|---|---|---|
| `v1_padded_width_divergence_is_column_padding` → `v1_372_bit_exact_to_fold_at_every_width` | assert the paths DIFFER at non-tight widths, bounded | assert they are BIT-IDENTICAL at all 19 geometries incl. h=93 |
| `folded720_v1_basic_matches_v1_path` | `expect_bit_exact` flag; 127/200 in a "divergence class" | flag deleted; every geometry asserts bit-exactness |
| `folded720_v1_pools_match_v1_path` | same | same |
| *(new)* `pyramid_stride_has_no_phantom_columns` | — | pins the decision at its owner over 24 widths |

**Prior-era artifacts** (registered in `benchmarks/eval_annotations.json` as
`v1-372-era2-phantom-column-pooling`, `v1-golden-tight-width-blind-pre-era3`,
`eval-root-2026-08-30-372-prior-era`): every 372 table extracted through the
buffered extractor at a **non-tight** width, which includes the day-old
`2026-08-30-full-features-372` root and the 2026-05-15 STORED root. **No 944
table and no 944-trained model is affected.**

**NOT flipped, listed for the user:** `zensim_validate::eval_roots::
DEFAULT_FEATURES_ROOT_372` stays pointing at the era-2 root until the
era-3 re-extraction lands and the shipped-B-under-C delta is reviewed.

### §3.31 — the era-3 eval root, and shipped B priced under C (2026-08-30)

**Built.** `/mnt/v/zen/zensim-training/2026-08-30-era3-full-features-372` —
eight corpora genuinely re-extracted under option C (cid22 4292, kadid 10125,
tid 3000, pipal 21800, konjnd 1008, aic3 600, csiq 866, live 779) plus the
`kon504` fresh subset. **Six are copied and remain prior-era**: aic4,
nonphoto, imazen26, sdr25, hfnlproxy, hf_nearlossless (registered as
`eval-root-era3-2026-08-30-mixed-era-copies`).

**A silent-loss bug in the build script, found and fixed.** `zv cid22` pointed
at the dataset PARENT rather than `CID22_validation_set`. The failure reports
`4292 pairs in 0.0s (12427864/s)` — no work done — then `0 valid pairs`, writes
a **34-byte empty cache**, and **exits 0**. The era-2 build hit the same thing
(its cid22 cache is also 34 bytes, and its CSV is timestamped a minute later
from a hand re-run). Now fixed with the symptom documented at the call site.

**The era predicate validated itself, for free.** Seven of eight era-2 vs
era-3 CSVs differ; **`pipal` is byte-identical**, because its images are
288×288 and 288 is stride-invariant — it sits entirely in the tight class where
C is a no-op. That is a control nobody had to build, landing exactly where
§3.30's predicate says it must.

**Shipped B under C** (same bake, same corpora, only the features root
changed — `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07`):

| corpus | era-2 | era-3 (C) | delta | n |
|---|---:|---:|---:|---:|
| cid22 | 0.882117 | 0.882141 | **+0.000024** | 4292 |
| konjnd | −0.649669 | −0.654272 | −0.004603 | 1008 |
| tid | 0.778520 | 0.778883 | +0.000363 | 3000 |
| csiq | 0.934208 | 0.934929 | +0.000721 | 866 |
| live | 0.897026 | 0.898517 | +0.001491 | 779 |
| kadid | 0.808474 | 0.808505 | +0.000032 | 10125 |
| aic3 | 0.765012 | 0.763666 | −0.001347 | 600 |
| **pipal** | 0.564971 | 0.564971 | **+0.000000** | 21800 |

**C does not meaningfully move B anywhere.** Largest is konjnd 0.0046, and
konjnd is read as a magnitude so that is B improving. cid22 moves +0.000024.
`pipal` at exactly zero is the instrument confirming itself. The features
change materially at non-tight widths (up to 81.6 % on a pool slot); the
pooled rank statistics barely move, which is what removing ≤16 columns of ~576
should do.

**NOT FLIPPED — the user's item.** `eval_roots::DEFAULT_FEATURES_ROOT_372`
still points at the era-2 root. The flip is now supported by the table above;
the caveat to weigh is that six corpora in the era-3 root are copied prior-era
rows, so flipping makes the default root era-mixed until those can be
re-extracted.


## §3.29 — the AVIF autotune training view: what we thought we were building vs what the data allowed (2026-09-04)

**Thought-why.** Union every scored AVIF DOE wave into one view, train a picker
that predicts `{backend, knob tuple, chroma, bit depth, speed}` plus expected
`{bytes, wall-time}` from source features + a target quality, and validate it on
held-out ODD origins.

**Actual-why, four ways the data said otherwise — every one MEASURED, not
inferred:**

1. **There are no odd origins, and there cannot be.** The 32 AVIF DOE references
   were k-means-selected under `--parity 0` precisely so no val/test-origin
   content could reach a training artifact. `origin_split.split_of` returns
   `train` for all 32, so the canonical validate/test buckets are structurally
   empty and `train_hybrid.py` correctly refuses the corpus with *"0 validation
   rows … not a train-biased even-only set"*. The fix was NOT to weaken that
   guard: `train_hybrid` gained a declared `SPLIT_RULE` hook implementing the
   already-registered even-only sub-split (`DATA_SPLITS.md` L158, the
   `avifgen-2026-08-06` precedent), which subdivides the TRAIN bucket only and
   hard-errors on any origin that is not canonical-train. 26 train / 6 `eval8`.
   **`eval8` is a leg-side holdout; the canonical one does not exist for AVIF.**
2. **`chroma` is not an axis.** Backend and chroma are perfectly collinear in
   every row (svt 4:2:0, zenrav1e 4:4:4; 1,114 `av1C` boxes, zero exceptions),
   because no chroma knob is wired for AVIF at all. It ships as a DERIVED
   attribute of backend. A picker fitted here learns the pair, and no
   re-analysis of existing bytes can split them.
3. **`wall-time` is a model of a model.** No fleet path persists a duration for
   any DOE cell, so the view's `encode_ms` comes from the speed instrument's
   `alpha + beta*MP` fits — single-threaded, q45-anchored, per-source fits on 5
   of 32 sources, pooled fits flagged `linear_model_failed` on 20/20 arms with
   beta spreading 24.3x. The trained time head fails the trainer's own
   `TIME_HEAD_R2` gate on both bakes; the shipped answer is the LUT, and even
   that is labelled modelled.
4. **The backend head does not work.** 54.0 % agreement with the measured
   per-image winner against a **67.7 % always-`zenav1-svt` baseline** — worse
   than the constant, and it never recovers a zenrav1e win (0/25, 3/27). The
   mechanism is legible in the corpus: the entire zenrav1e arm ran on the 1024²
   budget corpus, so the model sees that backend at exactly one size.

**What DID work.** The bytes/knob head: held-out mean regret 13.4 % (48-cell
cross-size-verified bake) / 14.5 % (143-cell full bake) against the per-row
oracle, and on screen content the core bake is **0.7 % mean, exactly optimal on
92 % of decisions**. Both bakes carry `safety_report.passed = false` and were
baked `--allow-unsafe`; those thresholds assume a corpus two orders of magnitude
larger than 32 references, and the violations are recorded rather than tuned
away.

**Two corrections made inside this pass, before publishing.** (a) The per-image
backend reference is a BUDGET-corpus verdict (`pixels` reads 1,048,576 for every
cropped ref); an earlier pass scored all 293 decisions and read 58.0 %, which
applied a crop verdict to native pixels — the scoped 161-row read is 54.0 % with
132 rows counted NOT-COMPARABLE. (b) `build_commit` came back `null` because a
`jj workspace` has no `.git`; it now falls back to the colocated primary repo.

**Three era-scoped facts were re-derived from the bytes rather than inherited:**
cross-era byte identity (12,000/12,000 shared cell identities, 0 conflicts), the
dead-knob census (which settles a contradiction between the Stage-A and
era-delta records — `scm3`/`tn0` inert at speeds 4 and 6, `scm3` LIVE at speed
7), and svt speed-dial aliasing (presets 8/9/10 ≡ 7).

Record: `zenmetrics/benchmarks/avif_autotune_v1_2026-09-04.md`. Contract:
`zenmetrics/benchmarks/avif_autotune_contract_2026-09-04.md`. Data:
`/mnt/v/zen/avif-autotune-2026-09-04/` (triple-mirrored;
`~/work/zen/DATA_PROVENANCE.md`).

---

### §3.32 — shipped B's training legs: the re-extraction is BLOCKED, and the defect is the DIAL (2026-09-04)

**Thought-why.** §3.27 established that the stored 372-col masked/IW block is pre-fix and
that shipped **B** "was fit AND calibrated on pre-fix masked/IW and is serving post-fix" —
"a genuine train/serve skew". §4c.3 of `benchmarks/v1_extractor_drift_2026-08-30.md`
registered the remedy as *re-extract B's ~227k training pairs and retrain*, priced at
"6–7 min of single-box CPU". The implicit model was: B is contaminated wholesale; a
re-extraction plus retrain repairs it.

**Actual-why.** Both halves of that model are wrong, measured.

1. **B is only ~13.5 % pre-fix by weight-fitting mass, but 100 % pre-fix in its dial
   anchor.** B = `0.8·cid + 0.2·kon`. The **cid head — 80 % of B — is fit on `hdr_v3mix`
   alone** (7,410 rows, extracted 2026-07-03, i.e. after both fixes), so it is already
   current-era. Only `kon` carries pre-fix legs, and within `canonhdr15` the post-fix
   `hdr_v3mix` is up-weighted ×15 = 32.7 % of that head's mass. Pre-fix share of the kon
   head 67.3 %; of B, **13.47 %**. Meanwhile `multiband_anchor_dial100.parquet` — B's
   *entire* dial calibration — joins into `safesyn` at **2,000/2,000** on
   `(ref_basename, f0)` and is therefore 100 % pre-fix. That is exactly the symptom shape
   §3.27 measured: rank barely moves (CID22 +0.0057) while the dial shifts −4.98/−5.86.
   The weights were never the problem.

2. **The re-extraction cannot be run** — ⚠ **numbers CORRECTED 2026-09-04, see §3.34.**
   `safesyn` (57.6 % of the kon head's mass, and the source of the anchor) was extracted
   from the `q<X>.png` decode cache, which is **0 % present** (0 of 3,000 sampled rows, all
   six codec families) — that part stands. The surviving bitstreams do re-decode to
   *different pixels*, but the magnitude first recorded here was measured through the
   third-party `image` crate, which reads an **XYB JPEG as an ordinary JPEG** and cannot
   decode AVIF or JXL at all. ~~worst `0.659 → 2875.0`, 69 % of cells over tolerance,
   roughly 10^4× the correction~~ — **RETRACTED.** Re-run with imazen decoders on all six
   families (360 rows, alignment gate 360/360): basic worst cell **5.481e+1**, the XYB
   family's own worst **30.31**, `|Δ| > 1.0` on **14 cells = 0.025 %**. The conclusion
   survives on a different number: re-decoding shifts shipped B's dial by **mean −3.658
   points, 73 % of the −4.98 era defect**, so a fresh safesyn still confounds the fix with
   a decoder-era term of comparable size. The compute estimate in §4c.3 was fine; **the
   inputs do not exist**, and that was never checked.

3. **A dial re-anchoring corrects the era term at ZERO rank cost.** `kadid`+`tid` exist in
   both eras with row-order-identical refs and byte-identical `human_score`, giving a
   matched-era anchor pair (13,005 rows, `target_score = max(ssim2_gpu,0)`; only the
   feature era differs). `bake_dial_refit shared-anchor` on each, scored on the current-era
   root: **SROCC identical to 5 dp on all five corpora** (spline is rank-invariant), while
   the per-pair dial moves **+6.196 (CID22, sd 2.246) / +6.235 (KonJND, sd 1.929)**,
   100 % of pairs > 0.5 pt. Against the defect's −4.977 (sd 2.301) / −5.857 (sd 2.299):
   same order, opposite sign, matching spread.
   **⚠ But anchor CONTENT moves the dial just as much** (−6.44 / −7.78 for a stored-era
   kadid+tid anchor vs the shipped safesyn one — a comparison itself confounded by
   procedure, `extend-top`/30 knots vs `shared-anchor`/12). So B's absolute dial is
   anchor-dependent at the same ±6–8 pt order as the era defect, and swapping anchors is
   not a drop-in fix.

**Also corrected here.** `zensim/CLAUDE.md` "Safe synthetic dataset" says *"the CSV
`decoded_path` PNGs no longer exist"*. That is true of
`2026-05-16/safesyn_with_iwssim.csv` — the actual extraction input — and **false** of
`synthetic-v2/training_safe_synthetic.csv`, whose `decoded_path` is the bitstream and is
present. The two files are row-identical and differ only in that column; conflating them is
what makes safesyn look re-extractable.

**Also found, not fixed.** `extract_features_372col` decodes via `image::open()` and
returns `None` on failure, so AVIF and JXL rows vanish **silently**: 240 of 360 probe rows
scored, i.e. 30.8 % of safesyn would be dropped without a word. Third instance of the
silent-row-drop class (`dataset_metric_baseline`, `zensim-validate --extract-only`,
now this). The decode helpers already exist at
`zensim-bench/examples/extract_features_372col_omni.rs:266-293` behind `extract-omni`.

4. **The fit chain is NOT byte-reproducible — and now the cost is priced.** Re-running
   `legs → gram → fit → ensemble → f16 bake → anchored bake` from the same stored legs
   breaks byte-identity in two named places: (a) `canonhdr15-bvls-raw` is an **iterative
   active-set (BVLS) solve**, so its `w` reproduces to **1.19e-5** relative against the
   closed-form lasso head's 2.25e-12, and 0.2× of that reaches the ensemble where one
   weight (`f83`) straddles an f16 tie — **371/372 f16 lanes identical**, bake sha
   differs; (b) the 823 B anchored-bake step's exact invocation is not recovered by
   `bake_dial_refit shared-anchor` at its defaults (`88a57447` vs the committed
   `7b326ac5`; the producer `shared_anchor_refit.py` was deleted 2026-07-29). The
   campaign's *"44/44 refits byte-identical"* determinism claim holds for a re-run **from
   cached grams**; it does not survive re-accumulating the grams from parquet.
   **Priced end-to-end**: rank reproduces to **≤ 3e-5 SROCC** (exactly 0.00000 on four of
   five corpora) and the dial to **≤ 0.071 points**, of which only ≤ 0.013 is the fit
   re-run itself — i.e. the pipeline's own noise floor is **~70× below** the −4.98-point
   defect. A fresh-legs comparison is therefore interpretable *functionally*, never on
   bytes, and any retrain moving the dial < ~0.1 pt has measured nothing.
   (Also: `ens-Pline-cid80.npz` was never emitted by the committed `cmd_ensemble`, which
   only produces `Pline-cid{30,50,70}` — a missing commit, not a lost recipe; the
   arithmetic reproduces the stored npz to 1.4e-14 with an exact bias.)

5. **The executable subset was RUN, and it falsified this lane's own registered prior.**
   §5 of the record predicted the launchable partial — kadid + tid swapped to current-era
   features, **1.94 % of the kon head's weighted mass, 0.39 % of B's** — was "very unlikely
   to clear" the 0.071-pt floor. Against a **matched control** (same pipeline, same day,
   stored legs), it clears it by ~13×: the dial moves **+0.838 (CID22) / +0.920 (KonJND)**
   mean, 99.5 % of pairs above floor, **in the correcting direction**, recovering 17 %/16 %
   of the defect's magnitude from 1.94 % of the mass; and rank moves on **all five**
   corpora — CID22 **−0.00087**, KonJND **+0.01244 |SROCC|**, AIC-3 +0.00106, TID +0.00156,
   KADID +0.00515 (⚠ KADID is B's train==val corpus *and* one of the swapped legs, so its
   gain is partly memorization, not generalization; the two genuine holdouts, CID22 and
   AIC-3, **disagree in sign**). Mechanism: BVLS is an **active-set** solve, and a
   1.94 %-mass perturbation moved three features (285, 336, 357 — all masked/IW) across the
   active boundary, `max |Δw|` 0.299 (89 % rel), support 85 → 86.
   **Do NOT extrapolate to the full re-extraction** — the effect runs through a non-linear
   active-set boundary, and the remaining 98 % is safesyn + cid22_train with safesyn
   BLOCKED. The arm is an instrument, not a ship candidate.

**Nothing was rewritten, retrained, or flipped.** Record:
`benchmarks/b_reextract_wave_2026-09-04.md`. Instruments + `_MANIFEST.json`:
`/mnt/v/output/zensim/bfresh-2026-09-04/`.

### §3.33 — class-C free slots: no new root was needed, and the free-40's route parity does not hold on real pixels (2026-09-04)

**Thought-why.** The class-C lane was scoped to emit the next tranche of "already in a
register, never emitted" 944 slots from a v1-basic-only walk, and then to **extract a new
dated fast-class root** (`/mnt/v/zen/zensim-training/2026-09-04-fastclass-classC/`) on the
fleet so the distillation lane would have tables carrying them. The implicit model was:
new slots ⇒ new columns ⇒ new extraction.

**Actual-why.** The model is wrong for this tranche, and measuring it first saved a fleet
wave.

1. **The class-C slots are EXISTING 944 positions, so every non-folded 944 root already
   carries them.** The 24 landed slots are the v2-348 `MSE` cell per (scale, channel) and
   the append `LUM_{DARK,MID,BRIGHT}_ERR` trio at Y per scale — positions a FULL 944 walk
   has always computed. What the class-C route changes is the COST of producing them, not
   their existence. MEASURED on
   `/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30/ext_cid22val.parquet` (regime
   `folded720append2pools`, `build_commit ced6f52a`): **4,292/4,292 rows non-zero on every
   one of the 24**. **No new root was built.** The distillation lane trains on the existing
   root with `scripts/sota944/slice_basic156_free_classc.txt` (289 coordinates).

2. **Real-pixel route parity: the class-C 24 hold, the previous lane's free-40 do not.**
   `v2_ab_extract` gained mode `foldapp2fast` (the cheap route) beside `foldapp2pools` (the
   full walk that built the root); both were run over the SAME 773 real pairs (1-in-9
   stride of `pairs_imazen26_png.tsv`, real zenavif/zenwebp/zenjpeg output, 64×48 …
   1024×1024). Against the free-features lane's own 2e-5 bar:

   | set | cells | over bar | worst \|Δ\| |
   |---|--:|--:|--:|
   | class-C (24) | 18,552 | **0 (0.00 %)** | **9.81e-8** |
   | free-40 (raw moments) | 28,601 | **2,607 (9.12 %)** | **3.63e-3** |

   The misses are entirely `GLOBAL_CLOSS` (1,467) + `GLOBAL_CGAIN` (1,132), with
   `GLOBAL_DMEAN` 8 and `LUMA_MEAN_REF` **0** — i.e. exactly the slots
   `global_stats_from_raw_moments` derives through `Σs²/n − (Σs/n)²`, a
   catastrophic-cancellation form, where the two routes' f32→f64 staging differs and the
   true value on a near-identical pair sits at the f32 accumulation floor (worst relative
   error ~55×). It grows with plane size (3.63e-3 in the largest-25 % bucket vs 1.60e-3 in
   the smallest). The previous lane's gate is synthetic-image-only, which is why it read
   5.35e-6 and passed. **Reported, NOT fixed — those slots belong to that lane.** Consequence
   for anyone shipping a fast-class model: it would be trained on one route's
   `GLOBAL_CGAIN`/`GLOBAL_CLOSS` and served the other's, disagreeing by up to 3.6e-3.
   Basic + peaks are BIT-identical between the routes (worst \|Δ\| exactly 0.0 over
   773 × 228 cells).

3. **A v1-only 944 walk does not leave every unreached slot at zero.** All twelve
   `PJND_FRAGILITY` slots (`f393 422 451 / 480 509 538 / 567 596 625 / 654 683 712`) read a
   constant **1.0** on 773/773 rows — a `finish_channel_scale` formula artifact on zeroed
   accumulators, pre-existing and identical on the `RawMoments` route. A training lane must
   slice to the free set explicitly; "every non-zero column" hands a model twelve constant
   columns that do not exist in the stored 944 tables.

**Cost of the tranche, published honestly:** +1.3–1.5 % of the 156 walk at 1T on the
native AVX-512 tier and +2.0–2.3 % on the v3/AVX2 tier (CI-excludes-1.0 at every 1T cell,
both tiers); threaded cells mostly straddle 1.0 and are reported, not asserted. The lane
brief expected "~zero marginal"; the measurement is what is published.

Record: `benchmarks/free_features_classC_2026-09-04.md`. Artifacts:
`/mnt/v/output/zensim/classc-2026-09-04/{native,capv3,routeparity}/`. Code: `a8b24c8e`.

### §3.34 — the safesyn re-decode probe, run with OUR decoders: a 52× retraction, and decoder era is a −3.66-point term on every corpus (2026-09-04)

**Thought-why.** §3.32 point 2 concluded that safesyn is not re-extractable because
re-decoding its surviving bitstreams moves the basic `f0..155` block by *"roughly 10^4×"*
the correction the wave existed to apply — worst cell `0.659 → 2875.0`, 69 % of cells over
golden tolerance. The implicit model was: the stored pixels are unrecoverable by a wide
margin, so the whole question is closed.

**Actual-why.** That probe decoded with the third-party **`image` crate**, in a
measurement whose purpose was tuning an imazen model — an IMAZEN-ONLY rule violation
(`~/work/zen/CLAUDE.md`), and the violation *is* the result. `image` reads an **XYB JPEG
as an ordinary YCbCr JPEG** and never applies the inverse XYB→sRGB transform, so the
`zenjpeg-420-xyb-e2` family (14.4 % of safesyn) was compared in the wrong colour space;
and it has **no AVIF and no JXL decoder at all**, so `zenavif-s5-e6` (34,001 rows) +
`zenjxl-e7` (26,362 rows) = **30.8 % of safesyn** was silently dropped and never measured.

Re-run through `zencodec` magic-byte detection + zenjpeg / zenpng / zenwebp / zenavif /
zenjxl — **360 rows, 60 per family, all six families, q5..q100 across 16 q values,
alignment gate 360/360 on `(ref_basename, cpu_ssimulacra2)`**, extraction
`360/360 scored, 0 failed`:

| quantity | retracted (`image`) | **corrected (imazen)** | ratio |
|---|---:|---:|---:|
| basic `f0..155` max abs | 2.874e+3 | **5.481e+1** | **52×** |
| XYB family worst cell | 0.659 → 2875.0 | **29.84 → 60.16** | **95×** |
| peaks / masked / IW max abs | 1.613 / 1.034 / 1.246 | **0.166 / 0.0411 / 0.132** | 9.7× / 25× / 9.4× |
| basic cells with \|Δ\| > 1.0 | not reported | **14 of 56,160 (0.025 %), 6 of 360 rows** | — |
| families measurable | 4 of 6 | **6 of 6** | — |

The corrected numbers land **inside the drift bounds `zensim/CLAUDE.md` had already
recorded on 2026-06-22** from an independent measurement (plain JPEG `max_abs ≤ 5` —
measured ≤ 0.48; XYB `≤ 42` — measured 30.31; JXL differs by decoder lineage — measured
the worst family at 54.81). The retracted 2,875 was outside all three, which is the tell.

**The conclusion survives, on a number that is now the interesting one.** Forwarding both
matrices through **shipped B** on the canonical runtime: re-decoding shifts the dial by
**mean −3.658 points (median −3.181, sd 2.589, 94.4 % of rows > 0.5, max 16.0)** against
an era defect of **−4.977 / −5.857**. **Decoder era is 73 % of the extractor era, same
sign.** So a fresh safesyn still cannot isolate the extractor fix — but as a confound of
*comparable* size, not of four orders. Median basic cell moves 1.29e-5; the retracted
doc's own median (1.09e-5) was the part that was never wrong. **What was wrong was the
tail, and the tail was the argument.**

**What this changes for every corpus, not just safesyn.** Any corpus whose pixels were
decoded once and stored is pinned to that decoder's era, and re-reading it today costs
~3.7 dial points *before* any model change. imazen-26 is not exempt. The remedy is not to
hunt for a corpus that escaped decoder drift; it is to choose a **deliberate** era,
re-extract through it, and record the decoder per format in the manifest — which is what
the imazen-26 anchor build (§3.35) does.

**Owner fix that made the probe possible at all.** `extract_features_372col` decoded with
`image::open(..).ok()?`: a third-party decoder behind a `?` that turned failure into a
dropped row. It now goes through `zensim-bench/examples/shared/zen_decode.rs` (one owner,
shared with `verify_bitstream_decode`) and returns `Result`; a row that cannot be decoded
aborts the run unless the caller passes `--allow-failures N` (default 0). First contact
found a further gap the pre-existing `extract-omni` helper shared: **every probed
`zenavif-s5-e6` row decodes to `Rgb16`**, which both helpers rejected — now flattened by
the canonical `zenpixels_convert::RowConverter`, not a hand-rolled `v >> 8`.

Record: `benchmarks/safesyn_zencodec_probe_2026-09-04.md`. Gate:
`zensim-bench/tests/zen_decode_formats.rs` (13 tests; the AVIF, JXL and XYB-JPEG ones are
failing-first against the old path). Artifacts:
`/mnt/v/output/zensim/im26anchor-2026-09-04/probe/`.

### §3.35 — the imazen-26 dial anchor, and B's dial split cleanly into ERA and CONTENT (2026-09-04)

**Thought-why.** §3.32 established that 100 % of shipped B's dial calibration comes from a
2,000-row safesyn subset, and §9d of `benchmarks/b_reextract_wave_2026-09-04.md` concluded
that swapping it was not a fix: *"B's absolute dial is anchor-dependent at the ±6–8 point
level, which is the same order as the era defect"*, so any re-anchor "trades one
uncontrolled ±6-point dial shift for another". The implicit model was that anchor CONTENT
and anchor ERA are comparably large, leaving no clean move.

**Actual-why.** §9d's estimate was **confounded** — it varied content (safesyn multiband →
kadid+tid) *and* procedure (30-knot `extend-top` → 12-knot `shared-anchor`) at once. Varied
cleanly, the two terms are an order of magnitude apart. The instrument is a 2×2 with a
matched middle arm: the **same 2,000 safesyn anchor rows, same targets, re-decoded and
re-extracted today**, sitting between the shipped anchor and a new imazen-26 anchor.

| term | held fixed | CID22 | KonJND | AIC-3 |
|---|---|---:|---:|---:|
| procedure floor | anchor + era; chain rebuilt | +0.031 | +0.028 | +0.028 |
| **ERA** | **content exactly fixed** | **+3.892** | **+4.798** | **+3.864** |
| **CONTENT** | era fixed, both current | **−0.395** | **−0.989** | **−0.233** |
| total | — | +3.528 | +3.837 | +3.659 |

**Era is 4–10× content on every holdout, and rank is identical to 5 dp across all five
arms on all five corpora** (CID22 0.88212, KonJND −0.51938, AIC-3 0.76501, TID 0.77852,
KADID 0.80847). TID and KADID show large positive content terms (+2.0/+2.3) — they are the
corpora B's kon head was fit on, KADID being its documented train==val corpus, so that is a
train/serve corpus shift, not a generalization signal.

A `shared-anchor` refit on the same content read today recovers **78 % (CID22) / 82 %
(KonJND)** of the −4.977 / −5.857 era defect, in the correcting direction, at **126× the
procedure floor** and zero rank cost.

**A design property worth writing down: `extend-top` alone cannot fix an era skew.** It
keeps the bottom and in-distribution knots VERBATIM and only extends above the top knot;
CID22's dial tops out at 90.41, below that domain. Swapping only the `extend-top` anchor
moved the human corpora by **0.000**. Only `shared-anchor` refits the in-distribution
spline, which is where the skew lives.

**The new anchor.**
`/mnt/v/zen/zensim-training/2026-09-04-imazen26-anchor-372/imazen26_multiband_anchor_dial100_2026-09-04.parquet`
(sha256 `b2e8ead6…`, 4,000 rows × 382 cols; LAN
`s3://zentrain/anchors/2026-09-04-imazen26-anchor-372/`). Built from **imazen-26 bigcodec
TRAIN encodes**, 4 lossy codecs × 10 decile bands × 100 rows, 192 distinct origins, 1,224
distinct refs, `target_score = max(score_ssim2, 0)` — the shipped anchor's own rule.
**Nothing was re-encoded**: distorted bytes were byte-range-read from the canonical run
tars, reference bytes are the local `clean-picker-corpus-2026-06-26` renditions (0 of 1,224
missing). It doubles the shipped anchor's rows and adds a **codec axis the shipped anchor
does not have at all** (that file has no codec and no q column).

**The 924 tables could not supply it.** Their `f156..f371` are STRUCTURAL ZEROS (0 of 5,000
sampled rows nonzero); shipped B reads 49 of those 216 slots, so an anchor cut from those
columns would feed it real zeros — the `--regime 944` mis-scoring hazard in another
costume. Every feature is freshly extracted at 372.

**DECODER ERA IS NOW A RECORDED PROPERTY.** `_MANIFEST.json` names the decoder crate,
version and commit **per format** (zenpng 0.2.0 `00d6deb`, zenjpeg 0.9.0 `fad6a0af`,
zenwebp 0.5.0 `20898b7`, zenavif 0.2.0 `6dfdf6f`, zenjxl 0.3.0 `f0efd6d`), because §3.34
measured that re-reading a stored-pixel corpus costs ~3.7 dial points. **Known
mixed-era caveat, stated rather than hidden:** `target_score` is bigcodec's stored ssim2
from *its* decode era while the features are decoded today. The shipped safesyn anchor has
the same shape of property, so this is not a regression — but it is not single-era either.

**Two owner corrections this forced.** (1)
`scripts/canonical_corpus/resolve_bigcodec_pair_uris.py` listed `zenjpeg_lossy` and
`zenwebp_lossy` as fetch-mode `object`; re-measured, `canonical/2026-06-27/<ds>/encodes/`
is **empty for all four lossy datasets**, so both are now `tarrange` — as written, every
zenjpeg/zenwebp `dist_uri` 404s, which fails as a *fetch* error and slips past that
script's own 100 %-resolution gate. (2) `fetch_bigcodec_bytes.py` hard-required a
`human_score` column and now auto-detects (`--score-col`), and carries a numeric
`row_index` through so a corpus cut can rejoin its rows.

**NOT SHIPPED, and the honest reason.** All gates pass — monotonicity **0.9770** (better
than shipped's 0.9740), tied 0.0000, G-RANGE PASS — but the dial **compresses**: reach
96.85 → 85.74, p5 13.73 → 22.91, dynamic range −10.5. About half of that is already in the
current-era safesyn arm (reach 94.23), so it is part era, part content. A Profile-B swap is
a ship-default flip and belongs to the user.

**The registered top-densification was RUN, and it half-worked.**
`imazen26_anchor_topdense_2026-09-04.parquet` (sha `0d5d27a3eb5be4f9`, 3,785 rows: 300 per
decile to 90, then 800 in [90,95) plus every one of the 285 rows the corpus has above 95 —
1,085 above 90 against the uniform cut's 400) moves `extend-top`'s saturation `k`
1.325 → **1.885** and recovers **reach 85.74 → 88.96** and **p95 98.44 → 99.12**. But
**`p5` does not move (22.91 → 22.99)**. So the top-end loss was a top-density artifact and
is fixable there, while the **larger remaining term is the FLOOR**, which `shared-anchor`'s
percentile-edge fit sets and `extend-top` structurally cannot touch. A low-band
densification is the next lever — registered, not run.

**ARM 2 — replacing safesyn INSIDE the kon head — is a measured NEGATIVE.** The leg
`imazen26_konleg_40k_2026-09-04.parquet` (sha `d2bbb218914cd2de`; 40,000 imazen-26 TRAIN
rows, disjoint from the anchor, `human_score = score_ssim2/100` UNCLAMPED so its 2,781
negative rows survive as safesyn's do) enters `canonhdr15` at weight **196086/40000 =
4.90215**, so the mix's weighted mass is unchanged and only the leg's CONTENT differs.
Result, against a same-day control that is **byte-identical to the b_reextract lane's armC**
(`f08b3c8052e13e37`): **CID22 0.88212 → 0.85100 (−0.03112, ~1,000× the 3e-5 floor)** and
**AIC-3 −0.00321** — both genuine holdouts DOWN — while TID (+0.0157) and KADID (+0.0179),
the corpora the kon head is fit on, go up. Dial monotonicity 0.9740 → 0.9566.
**Mass-matching is not diversity-matching:** safesyn's leg spans 1,495 references and six
codec families (including two XYB-JPEG variants that exist nowhere else); the imazen-26 leg
spans **212 origins and four codecs**, up-weighted 4.9× to equal influence. The corpus's
212 train origins are a diversity ceiling, so more rows would not obviously fix it.
**Replacing safesyn is safe where it is a CALIBRATION input (the anchor: rank-invariant)
and expensive where it is a FITTING input (the kon head: −0.031 CID22).**

**Provenance correction to `b_reextract_wave_2026-09-04.md` §10a:** it describes the
uncommitted `ens-Pline-cid80` step as *"anchor-normalised 0.8\*cid + 0.2\*kon"* and omits
the **per-head taus**. Without them the reconstruction misses the stored npz by max |Δw|
**1.8e+2** (tau on both) or **1.5e+0** (tau on neither). The owner's `HEAD_POOL` has
cid `tau = 0.0`, kon `tau = 0.005`; with those it reproduces to **2.84e-14** with an
identical 95-feature support. This lane's arm-2 driver proves the reconstruction against
the stored artifact before using it.

Record: `benchmarks/imazen26_anchor_2026-09-04.md`. Artifacts:
`/mnt/v/output/zensim/im26anchor-2026-09-04/{build,arms,probe}/`.

### §3.36 — three 944-POOLS eval instruments built, and the 944 identity vector is NOT zero (2026-09-05)

**What was needed and did not exist.** Grading a bake that reads `f156..371` (any model
using the free set's 72 peaks) at 944 width required a dial grid, a negative-tail probe and
an identity probe all in the **`folded720append2pools`** era of
`/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30/` (`build_commit ced6f52a`). Only the
grid existed (`wlin7b-2026-08-30/dial_grid_944col_POOLS_2026-08-30.parquet`). **The two
stored 944 negative-tail probes are the wrong era** — `negtail_probe_944_2026-08-01era` and
`…_era2r4_foldapp2` both carry `f156..371` STRUCTURALLY ZERO, which is exactly the block a
peaks-using bake reads, so serving either would have mis-predicted silently. **No 944
identity probe existed at any era.**

**What was built** (all in `/mnt/v/output/zensim/dfree-2026-09-05/`, registered append-only
in `benchmarks/dial_addressability_floor_2026-09-04.json` as `(instrument, peer_ssim2)` rows):

| instrument | rows | sha256 | rule |
|---|--:|---|---|
| `probes/identity_probe_944pools_2026-09-05.parquet` | 39 | `31fc4403…` | the 39 dial-grid references paired with themselves, extracted by the PINNED pools-era binary |
| `probes/negtail_probe_944pools_2026-09-05.parquet` | 2000 | `bafc8994…` | `cut_gaddr_negtail_probe.py`'s registered rule (20 equal-count quantile bins over the negative population, lowest 100 row indices per bin) applied to the pools safesyn leg's `human_score × 100` |
| `anchors/anchor944_r1b_dial_{clamped,negrich}[_id21].parquet` | 1976 / 1997 | `235dd65c…` / `5742aa7b…` / `2871bb4e…` / `7af3e25e…` | stride 55 (the era-2 `anchor944_pools_dial` manifest's own rule) minus the 44 rows the negtail probe also selects, so anchor and probe are **disjoint by construction** |

**The era control, run before anything rested on it.**
`wlin7b-2026-08-30/v2_ab_extract_PREFIX_PINNED` at `ZENSIM_AB_MODE=foldapp2pools`
reproduces **both** the r1b root (12 `ext_imazen26` rows × 944 = 11,328 cells, max |Δ| 0.0)
**and** the POOLS dial grid (7 matched cells × 944 = 6,608, max |Δ| 0.0) **bit-exactly**.
Separately MEASURED and REFUSED: `ext944-era2r4-2026-09-01/` declares the same regime and
ships an `anchor944_pools_dial.parquet` and a pools-regime `ext_kadis.parquet`, but is a
**different extractor era** — same rows, same order, `human_score` bit-identical, features
NOT (max |Δ| 7.46e-2 on `f0`, 1.96e-2 on `f160`, 3.13e-3 on `f930`).

**The finding: at 944 width the identity feature vector is NOT the zero vector.**
`dial_addressability.rs` carries a named constant asserting *"ref == dist yields all-zero
features for every image; identity dial = dial(0-vector)"*, and the 372 probe's registry row
records that as a measured property. On the 39-row 944-POOLS probe, **286 of 944 slots are
non-zero on at least one row and they VARY BY IMAGE** — the fold's own basic block
(`f0`/`f1`/`f2` at 4.7e-6 … 8.1e-4), every peak triple, the whole `PJND_FRAGILITY` family
(the formula artifact `free_features_classC_2026-09-04.md` §6.3 documents), and
`LUMA_MEAN_REF`, which is REFERENCE-ONLY and therefore cannot be zero on a ref==dist pair.
C5 and C6 still measure correctly (`IdentityMeasure` counts rows outside the band and
compares grid cells against `dial_max`), but **the note string is wrong at this width and is
emitted verbatim into every `--gaddr-json`**; the new registry row states the correction in
its `measured_property` field. Making the constant conditional on the probe is REGISTERED,
NOT DONE.

**Round-trip control:** `peer_ssim2` graded against its own three freshly-appended rows
reads **SHIPPABLE, 15 pass / 0 fail / 0 not-measured**, every bar tied bit-exactly; and the
four grid scalars (`min −55.354544, max 99.993468, p5 12.0158692, p95 95.4858354`) reproduce
the parked `gaddrinst` lane's independently-computed full-944 values EXACTLY. ssim2's
identity is **100.000000** on all 39 (38 carried from the 372 lane's `zenmetrics batch
--metric ssim2` run; the 39th, `9059ec43b26aa167_769x513`, measured here by the same tool).

**⚠ These bars are NOT the 372 bars.** Most sharply on A8: this probe's ssim2 p1 is
**−64.23** against the 372 kadis probe's **−187.13**, because safesyn's negative population
is shallower than KADIS's. A number from the two instrument sets must never be compared.

Record: `benchmarks/d_free_id100_2026-09-05.md` §§1.4, 2, 3. Artifacts + shas:
`benchmarks/d_free_id100_bakes_2026-09-05.pointer.md`.

### §3.37 — the v1-372 eval roots predate option C, so every 372 verdict is one era behind the runtime (2026-09-05)

**Thought-why:** the ship-flip lane needed to know which extraction era the *product
runtime* emits, so it could say whether a candidate bake's published numbers describe what
users get. The expectation, from the roots' own manifests and from `CLAUDE.md`, was
"HEAD == the 2026-08-30 root": that root is the DEFAULT `--regime 372` features root, it
was built the same week, and `CLAUDE.md`'s EXTRACTION PERF section said option C — the one
known pending v1 semantics change — was *"**Not flipped** — default untouched pending the
era rollout"*.

**Actual-why:** option C has been flipped since `56bbcda2` (2026-08-30 **15:43**), whose own
message calls it *"STAGE 1 of the C rollout"*. The default root was built at `ea16c7ee`,
2026-08-30 **13:21** — **two hours earlier**. `CLAUDE.md` was stale by six days.

**MEASURED.** `extract_features_372col --corpus pairs-tsv --path
/mnt/v/dataset/csiq/csiq_pairs.tsv` at HEAD — the *same tool* on the *same input file*
`scripts/canonical_corpus/build_eval372_root.sh` used — against the root's own build output
`~/tmp/eval372root/csiq.csv`. Row alignment established first, not assumed: `ref_basename`
order identical and `human_score` **bit-identical positionally on all 866 rows**.

| block | cells differing | max \|Δ\| |
|---|--:|--:|
| basic `f0..155` | **120,804 / 135,096** | **4.536785** |
| peaks `f156..227` | 34,566 / 62,352 | 0.326375 |
| masked `f228..299` | 62,346 / 62,352 | 0.067955 |
| iw `f300..371` | 62,346 / 62,352 | 0.079387 |

Identical results against BOTH v1-372 roots (`2026-08-30-full-features-372` and
`2026-05-15-full-features`) — consistent, since those two agree on basic+peaks. Every row
differs, on 285–341 of 372 slots; no row and no slot is clean. CSIQ is 512×512, exactly the
padded-width divergence class C removed (the commit: *"it put 512/576/768/1024/1152/2304
all in the divergent class"*).

**Two alternatives ruled out by measurement, not by argument.** `ZENSIM_ERA2_DENSE=0`
reproduces HEAD's output **byte-for-byte**, so `515001dc`'s era-2 dense flip is not the
cause (it moves only `f372+`). `v1_golden_bytes` passes **5/5** at HEAD, because every
golden fixture is 64×64 / 96×96 / 128×128 / 200×150 — tight class or below the tile — which
is the same blindness `CLAUDE.md` had already flagged for the goldens and which the C
rollout's own non-tight fixture only partly covers.

**What this does and does not invalidate.** It does **not** touch a same-root A/B (both arms
read the same bytes) and it does not reorder 372 cells relative to each other. It does mean
every ABSOLUTE `--regime 372` number — shipped D's CID22 **0.86338** included — is an era-1
read of a model the product now feeds era-3 features. **The 944 roots are NOT affected**:
`56bbcda2` verified structurally (not just numerically) that the fold's production path
never references the padding owner, so the 944 regimes are unchanged by construction.

**Registered, not run:** re-extract the 372 root at HEAD
(`scripts/canonical_corpus/build_eval372_root.sh` + `pack_eval372_root.py` — a decode-bound
corpus pass, no code change) and re-verdict the D/B lineages on it. Registry:
`benchmarks/eval_annotations.json` → `v1-372-eval-roots-predate-option-c-2026-09-05`.
Record: `benchmarks/d_ship_flip_2026-09-05.md` §3.

---

### §3.38 — the RUNTIME-era 372 root exists; the canonical DIAL GRID's pixels do not (2026-09-05)

**Thought-why.** §3.37 registered "re-extract the 372 root at HEAD" as a decode-bound corpus
pass with no code change, and `benchmarks/d_ship_flip_2026-09-05.md` §5 registered the peaks
refit that would be gated on it. Both assumed the three G-ADDR instruments could be rebuilt
the same way.

**Actual-why.** The ROOT rebuilt cleanly. Two of the three instruments did too. The dial
grid could not, and the reason is a data loss nobody had connected to it.

**The root.** `/mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC`, era token
`v1postc`, `build_commit 4fbd8ff8`, `feature_set_id basic+peaks+masked+iw@w372/v1postc#d16a1091`,
named once in `zensim_validate::eval_roots::POSTC_FEATURES_ROOT_372`. Eight corpora
re-extracted at HEAD (cid22 4,292 · kadid 10,125 · tid 3,000 · csiq 866 · live 779 · pipal
21,800 · konjnd 1,008 · aic3 600), six byte-copied from the `v1cur` root and era-stamped per
file (aic4, nonphoto, imazen26, sdr25, hfnlproxy, hf_nearlossless), `kon504` derived plus its
side root. **`pack_eval372_root.py` now REFUSES** a fresh table whose `human_score` is not
BIT-identical positionally to the superseded root's — the assumption every era comparison
rests on, previously unchecked. It passed 8/8. It is **NOT the default**; which root a
flagless `bake_verdict` reads stays a user decision.

**And the era it exposes is mild on RANK.** MEASURED across five shipped 372-class bakes,
`v1cur` → `v1postc` moves CID22 SROCC by **≤ 6.8e-4** (shipped D −5e-5, previous D −5e-5,
Profile A −5.9e-4, shipped B +2e-5, the new peaks arm −6.8e-4). That is two orders below the
`v1pre` → `v1cur` step's |0.489| worst case (§3.28), and the reason is structural: that flip
moved `f228..371`, which only some models read, while this one moves `f0..227`, which every
372 model reads — but smoothly, so a rank statistic absorbs most of it. **The DIAL moves
much more**, so this is not licence to read a stored 372 dial number as current.

**⛔ The canonical dial grid is UNREBUILDABLE.** `dial_grid_372col_2026-05-29_quarantined_v2`'s
own build list (`eval_panels_2026-05-29/qsweep_372_grid.tsv`) points every distorted cell at
`/mnt/v/input/zensim/images/<ref>/<codec>/q<N>.png` — the decode cache deleted 2026-06-22 —
and **0 of its 2,560 rows' paths exist**. Its jpeg leg was `mozjpeg-rs-420-e4`; the surviving
`dial-grid-pixels-2026-07-27/` set is a `zenjpeg` re-encode of the same (image, codec, q)
lattice. Confirmed numerically before it was confirmed from the build list: a PRE-option-C
binary (`27cfde15`) does not reproduce the stored grid from those pixels either — basic
max |Δ| **0.703** on 2,168 of 2,340 control cells, against **0.055** for the entire
preC→HEAD era step.

**The surviving pixels are the right instrument anyway, and this is the sharp part.** The
registered `peer_ssim2` G-ADDR grid pins were measured ON the 2026-07-27 pixels:
`ssim2-bar-2026-08-31/dialcells_ssim2_qv2grid.tsv` **is**
`dialcells_ssim2_944grid.tsv` restricted to these 4,424 keys (4,424 of 4,424 exactly equal),
and the 944 grids were built from them. Spot-check through `zenmetrics batch --metric ssim2`
on the new pixels: `(00b13be94a4867dd_1022x818, jpeg, q0)` reads **−8.0345** against the pin
table's **−8.0450**. So the canonical instrument has ALWAYS paired ssim2 truth from one pixel
set with features from another; the rebuild removes that, it does not introduce it. Derived
through the owner, `peer_ssim2` on the new grids reproduces its canonical registry row on all
eight scalars (min −55.354544 · max 98.376644 · p5 10.26332105 · p95 95.45929934999998 ·
reach 153.731188 · DR 85.19597829999998 · mono 0.99235757295044 · tied 0.0000).

**The identity probe is ERA-INVARIANT**: a HEAD rebuild of the same 38 `ref == dist` pairs is
**byte-identical** to the registered 2026-09-04 probe (sha `e6f9096b8e0ebd97…`, 0 nonzero
cells of 14,136). At w372 the identity feature vector is the zero vector in both eras — which
does NOT extend to w944 (§3.36 measured 190 of 944 slots non-zero there).

**The negative-tail probe IS recoverable, and its provenance is now pinned.** All 2,000 rows
join UNIQUELY (0 ambiguous) to `kadis700k_canonical_gpu_2026-07-01.parquet` on
`(feat_0, feat_1, feat_2, score_ssim2_gpu)` → `distorted_url` (R2) + `source_filename` (local,
2,000/2,000). Control: a PRE-C extraction of the recovered pixels reproduces the stored probe
to ≤ 7.66 absolute on a ~1.2e5-scale slot (6.2e-5 relative), while the preC→HEAD era step on
the same rows is 18,953 on that slot.

**The frozen safesyn Gram is `v1pre`, measured not assumed.**
`linear-probe/grams/safesyn.npz`'s first and second moments match
`canonical-2026-05-21/train/safesyn.parquet` exactly on every column checked (f0
`s=3688.568709 S=272.523676`; f1, f155, f156, f200 likewise, n = 196,086 both sides). Every
bake in the ADD156 / D lineage is therefore trained two extraction eras behind what the
runtime serves it.

**A silent-corruption class worth its own line:** `extract_features_372col` sorts its output
by `ref_basename` before writing (`:216`), so positional re-attachment of key columns is
UNSOUND for any input not already in that order — and it fails with the right row count and
every key present exactly once. It scrambled the first build of this grid across its 38
references (shipped D's ladder monotonicity 0.9847 → 0.5611). Carry a numeric `row_id` and
invert the permutation; do not change the sort, because the root build's positional
`human_score` gate depends on both sides having it.

Artifacts: `/mnt/v/output/zensim/dpeaks372-2026-09-05/`. Record:
`benchmarks/d_peaks_372_postC_2026-09-05.md`.

---

### §3.39 — the floor-dense LADDER instrument: jpeg's "three lowest settings" were one setting sampled three times (2026-09-05)

**Thought-why.** §3.38 rebuilt the dial grid at the runtime era from the surviving
2026-07-27 pixels and treated the instrument as fixed. The remaining ladder problem was
assumed to be extraction era.

**Actual-why.** It was the **encoder axis**. `A7r` asks whether a codec's three lowest
configurable settings still resolve, and on every grid built before this one that question
was unanswerable for `jpeg`: **zenjpeg emits ONE bitstream for every q in 0..10** —
identical bytes AND identical ssim2 to 6 dp, on every reference tested — so the grid's
q 0 / 5 / 10 are one setting sampled three times. The mentor's jpeg bar was therefore a
vacuous `0.0000`, which anything passes, and the incumbent's own grading shows it as jpeg
`bottom_medians` **22.22 / 22.22 / 22.22**.

**Per-codec floors, measured** (same setting == same encoded bytes AND same ssim2):

| codec | plateau | first DISTINCT |
|---|---|---|
| `zenjpeg` | q 0..10 — **eleven settings, one output** | q = 11 |
| `zenwebp` | q = 0 | q = 1 |
| `zenavif`/`zenravif` | q 0..1 | q = 2 |
| `zenavif`/`svt-rs` | q 0..1, then **pairwise ties** (quality 0..100 onto QP 0..63) | q = 2 |
| `zenjxl` | **distance >= 25** (26/30/40/50 byte-identical to 25.0) | d = 24 |

**So the rule is DEDUP BY ENCODE HASH, never a per-codec step table.** `avif-svt` is
**36.4 %** duplicate settings against `avif-rav1e`'s **3.0 %** on the same quality axis —
a 12x difference no fixed step could express. The instrument keeps DISTINCT settings only
(which is what lets `dial_addressability.rs` stay unchanged: its "bottom K" become the
bottom K *configurable* settings by construction); the archive keeps every step flagged.

**What it changed.** jpeg's mentor bar `0.0000` -> **0.5385**, and **shipped Profile D —
a clean A7r pass on every older grid — FAILS**, on jpeg, by one ladder (20/39 vs 21/39).
Profile B fails all five codecs with a POSITIVE `dial_min` on each. Nothing installed.

**And the fix cannot be calibration.** All 19 of D's failing jpeg ladders are inversions
in the RAW pre-spline model (raw-vs-dial ordering verdicts agree **39/39**), and the two
shipped D bakes — same ADD156 weights, different output spline — have **identical A7r on
all five codecs** while `avif-rav1e` `dial_min` moves −13.49 → −59.81. A monotone spline
moves range, never rank. **The A7r lever is the weights.**

**A codec defect this surfaced, filed as `imazen/jxl-encoder#101`:** at butteraugli
distance **>= 10.0 exactly** (9.9 is fine), the encoder writes a `SizeHeader` rounded UP
to even, so a 513x769 source declares 514x770 and cannot round-trip. Read from the
codestream's own header, so encoder-side. Pre-existing — the 2026-07-27 sweep shows the
identical signature on the same 13 odd-dimensioned images. It removes exactly those
ladders' FLOOR cells, so they are excluded as truncated-floor rather than graded several
steps up the curve; jxl carries 26 ladders, not 39. **It was diagnosable with no re-encode
only because this run persisted encoded bytes** — the discipline §3.38's own grid was
built in the absence of.

**Two process failures worth recording as data-integrity lessons.** (1) Editing a shell
script while bash is executing it: bash reads scripts incrementally by byte offset, so a
mid-run edit made it resume at a stale offset and die with `unexpected EOF` *after* the
avif leg and *before* jxl, with no COMPLETE marker and no error in the leg logs. Runs now
launch from a frozen copy. (2) Two long runs died silently when nohup'd inside a
backgrounded shell — rc=137 and rc=143, the latter at 786 s with 13.7 GiB still available,
so NOT memory: they were killed with their task's process group. Long jobs now launch
under `setsid`. Both failures are invisible in the output data; only the exit code and a
missing marker distinguish them from success.

Artifacts + provenance: `~/work/zen/DATA_PROVENANCE.md` ★ ladder-instrument-2026-09-05.
Record: `benchmarks/ladder_instrument_2026-09-05.md`.
