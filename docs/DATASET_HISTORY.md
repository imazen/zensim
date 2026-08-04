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
| **hdr_v3mix** | 7,410 HDR-JXL renditions; target = cvvdp-mix `0.5·clip01(ssim2)+0.5·clip01((JOD−6)/4)` | B cid head (80%), BHdr, B kon head, B winsor corpus | **HELPED — most load-bearing** | the cvvdp-mix TARGET is the driver (+0.039 CID22 vs pure ssim2); transfers to SDR linearly | `profile_b_methodology_2026-07-12.md §2`; `linear_projections_2026-07-03.md` (commit `1b2bdb9b`) |
| **safesyn** | 196,086 synthetic tiles (CLIC/CID22/kodak/gb82-sc crops), ssim2-derived target, CID22-49-purged | A (v11→v47), B kon head | **HELPED (foundational)** | base training mass; supplies the BVLS kon head its bulk | `CLAUDE.md` "Safe synthetic dataset"; `DATA_SPLITS.md §3` |
| **cid22_train** (201-ref) | 17,611 rows, **ssim2-anchored** (`human_score`=`ssim2_gpu/100`, NOT MCOS) | A, B kon head | **HELPED (train-legal)** | 201 refs disjoint from the 49-ref holdout; human MCOS never trains | `DATA_SPLITS.md §7.1` |
| **kadid / tid** | 10,125 / 3,000 rows, DMOS/MOS; ~95% analytic (non-compression) distortions | A, B kon head (low weight) | **NEUTRAL / integrity guard** | ssim2 tuned in-sample → never a scoreboard; train==val | `DATA_SPLITS.md §3` (T1) |
| **konjnd-dense** | 20,160 rows (active-mix target) + 1,008-ref val (PJND) | A (modest), B kon head (**≈wash**) | **HELPED (A) / NEUTRAL (B)** | for B the KonJND skill is the **BVLS sign-mask, not the corpus** (`canonkjhdr15`≈`canonhdr15`) | `linear_projections_2026-07-03.md` round-2 |
| **inclusive-winsor corpus** | 10,810 = hdr_v3mix + zenjxl near-lossless SDR sweep; fits B's winsor bounds only | B calibration | **HELPED (user-caught fix)** | frees 245/372 clamped-constant features → near-lossless dial 91.5→96.1 at ~0 MOS cost | `profile_b_methodology §3` (commit `aaa1ecac`) |
| **cvvdp_iwssim_LARGE** | 73,300 rows, `mix_cv40_iw60` target, **300 feats** | PreviewV0_5Balanced (V_22, not a current Profile) | **HELPED (V_22 era)** | +0.068 KADID/+0.087 TID; "marginal value is in content types not in training" | `v22_mix_LARGE_iwssim_methodology_2026-05-18.md` |
| **bigcodec / imazen-26** (canonical-picker) | 2.9–5.7M ssim2-normalized real-codec cells, 414 imazen-26 origins | MLP recipes (w≈0.25, later 0.5); **EXCLUDED from linear B** | **⚠ RE-FRAMED 2026-08-04 — the exclusion was RIGHT and its MLP carve-out was NEVER VALIDATED.** Original: "EXCLUDED from linear — poisons CID22 (0.65–0.76); MLPs absorb it via capacity" | a 372→1 linear head gets pulled to the ssim2 multi-codec regime. **"MLPs absorb it" was tested on CID22 ONLY. Measured 2026-08-04: at ~78% of row mass (bigcodec + its distillation teachers) the 944-era recipe collapses classic-IQA breadth AND KADID at EVERY feature width — 372-width arms on the 944 data score CSIQ 0.681/0.438, LIVE 0.666/0.300, KADID 0.546/0.641 vs B 0.934/0.897/0.809 and winner_dial 0.958/0.960/0.946. MLPs did not absorb it; they inherited it on the axes nobody re-checked.** Why it stayed invisible: KADID/TID were classified "integrity guard, never a scoreboard" (row above), and CSIQ/LIVE did not exist as corpora until `ac787382` (2026-07-18), AFTER the MLP recipes were set — so no gate in force at the time could fail. | `linear_projections_2026-07-03.md` Falsified #1; commit `c7900d53`; falsification: `sota944_campaign_2026-08-03.md` width-discriminator (k=2/arm — seed spread too wide to rank widths; the breadth/KADID collapse is robust to that limitation) |
| **KADIS-700k** | 700k analytic-distortion cells on 140k Pixabay refs; GPU-metric targets, no human labels | negative-region calibration; the §8.32-8.33 MLP-neg prototype (not shipped) | **EXCLUDED for shipped fit; diagnostic/negatives only** | cvvdp-scalar target falsified (V41); analytic + more-photographic bias | `DATA_SPLITS.md §2b`; `project_profile_b_hdr.md §8.31-8.33` |
| **kadis-hdr** | 11,400 PQ-domain synthetic HDR cells, 1,140 imazen-26 refs | BHdr breadth retrain | **EXCLUDED — falsified on UPIQ** | synthetic HDR breadth does not transfer to human-MOS HDR (linear family) | `bhdr_improvement_split_lineage §8.8-8.15` |
| **CID22-49 / AIC-3 / AIC-4 / UPIQ-HDR / JPEG-AI-SDR25** | human-MOS holdouts | **HOLDOUT-ONLY** | **EXCLUDED (sacred)** | the only held-out human generalization gates; UPIQ-380 now BURNED (~21 looks) | `CLAUDE.md` "CID22 is VALIDATION-ONLY"; `DATA_SPLITS.md §3` |

**Superseded / unverified-ship:** hdr_v3 (pure-ssim2, prior BHdr `anchored2`); e1-fill/zenjpeg-420-e1 (V0_7 line, built ~85%, **no promoted-bake trail** → UNVERIFIED); KonFiG-IQA (v53 group, **no shipped-bake trail** → UNVERIFIED); cvvdp-only LARGE (falsified, craters KonJND); dssim co-train (falsified, see §2). Pre-`VALIDATION-ONLY`-era weights (Nelder-Mead `ae28074`, CMA-ES) trained on {KADIK, CID22, TID} *including CID22* — any CID22 SROCC from that era is **contaminated**, archaeology only.

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

---

*Maintenance: update this doc whenever a verdict is added, re-framed, or a confound is found.
Prefer editing here in-place over a handoff note. Every claim keeps its citation.*
