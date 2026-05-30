# JXL training-data investigation — 2026-05-30

**Question:** what data do we need to add JXL to training (to fix the
near-lossless JXL dial underscoring found 2026-05-29)?

**Answer: none — the data already exists and was simply never trained on.**

## What exists

| artifact | status |
|---|---|
| JXL encodes — 3,288 sources × 16 q (q5…q100), `.jxl` + decoded `.png` | ✅ on disk `/mnt/v/input/zensim/images/<src>/zenjxl-e7/` |
| JXL scored — **cvvdp** (31,300) / **iwssim** (32,000) / **ssim2** (24,500) rows | ✅ `canonical-2026-05-21/scores/{cvvdp,iwssim,ssim2}_imazen*.parquet` |
| JXL feature-extracted, 372-feat, training-ready | ✅ `canonical-2026-05-21/features/cvvdp_iwssim_LARGE_372col.parquet` — **31,535 zenjxl rows** (of 73,300; also 32,454 jpeg / 6,507 avif / 1,590 png / 1,214 webp) |

Confirmed by joining LARGE's `iwssim` values back to the iwssim sidecar
(which retains the codec column LARGE dropped). All 31,535 JXL rows have
populated targets (`human_score`, `cvvdp_score`, `iwssim`; `ssim2_gpu` on 77%).

## The root cause

**v47 (Profile::A) — and every shipped bake — never trained on
`cvvdp_iwssim_LARGE`.** v47's recipe groups (per its methodology §g) are:
`safesyn + cid22_train + kadid + tid + konjnd_dense`. LARGE is not a
`--group`. So the JXL training data exists, fully prepared, and is unused.
That is why all bakes underscore near-lossless JXL — they never saw it.

(Note: LARGE's JXL coverage is **near-lossless-heavy** — iwssim p5=0.995,
cvvdp 8.9–10.0 — which is exactly the underscored region, so it directly
targets the dial finding.)

## Genuine remaining gaps (secondary)

1. **Low-q JXL (q5–q60)** is unscored. LARGE's JXL is near-lossless-only
   (iwssim ≥ 0.935). The q5–q60 encodes exist on disk but were never in the
   cvvdp/iwssim sweep that fed LARGE. Per the q5–q60-density rule, full-range
   JXL needs them scored + feature-extracted.
2. **butteraugli + dssim** are not scored for JXL anywhere (only cvvdp /
   iwssim / ssim2).

## Scale gotcha (V39 trap, avoided)

LARGE's `human_score` is on a **0–100** scale (mean 45) while every other
group is **~[0,1]** (ssim2/100). The trainer does NOT per-group normalize, so
adding LARGE raw would diverge (documented V39 learning). Fixed by
pre-normalizing LARGE's human_score ÷100 →
`/mnt/v/output/zensim/jxl_exp_2026-05-30/cvvdp_iwssim_LARGE_372col_hsnorm.parquet`.

## Experiment in flight

`zensim/weights/manifests/v47_plus_large_2026-05-30.toml` = the v47 recipe
+ a 6th group `cvvdp_large` (the hs-normalized LARGE, train_w 0.6). Tests
whether the existing JXL data closes the near-lossless JXL dial underscoring
without regressing held-out CID22 / AIC-3 / AIC-4. Result + verdict appended
below when training + the two-panel eval complete.

## RESULT — FALSIFIED (2026-05-30)

Adding LARGE as a 6th group made things **worse**, not better:

| metric | v47 | v47+LARGE | Δ |
|---|--:|--:|--:|
| **JXL dial @best** (the target) | 74.9 | **66.2** | **−8.7 (WORSE)** |
| CID22 SROCC (gold holdout) | 0.866 | 0.732 | **−0.134** |
| AIC-3 SROCC | 0.768 | 0.638 | **−0.130** |
| AIC-4 SROCC | 0.885 | 0.683 | **−0.202** |
| TID SROCC | 0.793 | 0.807 | +0.015 |
| KonJND SROCC | 0.419 | 0.435 | +0.017 |

Dial structure stayed clean (flat/clamp 0.000, monotonicity 0.977, G3 1.00),
but JXL reach **regressed** (74.9 → 66.2) and the three **compression
holdouts cratered** (−0.13 to −0.20) while the synthetic-distortion guards
(TID/KonJND) nudged up.

**Mechanism — the CVVDP-emulator dead-end (V41), reproduced.** LARGE's
`human_score` is the cvvdp/iwssim-derived target, NOT ssim2. Training toward
it pulls the metric toward CVVDP-shaped output, which CLAUDE.md already
documents as a dead end for human-MOS (CID22 0.66 vs 0.88). The
compression-down / synthetic-up split is its fingerprint. LARGE's
near-lossless skew (iwssim p5=0.995) compounds it — 73k high-q rows pull the
calibration toward the top, compressing the low/mid range CID22/AIC live in,
which is *also* why JXL near-lossless reach got worse.

**The recipe-gap hypothesis is FALSIFIED for LARGE-as-is.** The JXL data
exists, but in the WRONG SHAPE: cvvdp/iwssim target + near-lossless-only.

**Correct next experiment (the steelman, not yet run):** score the
**full-range** on-disk JXL (q5–q100, 3,288 src × 16 q) with **ssim2** (the
ship-recipe target) — and butteraugli/dssim for completeness — build a
balanced-quality, ssim2-targeted `jxl.parquet`, and retrain. That tests
"JXL in training, RIGHT target + RIGHT distribution" — distinct from this
falsified "LARGE-as-is" run. This is the "run all metrics on full-range JXL"
work; the falsification now concretely motivates it.

Experiment bake: `/mnt/v/output/zensim/bakes/v47_plus_large_2026-05-30.bin`
(28,334 B, best val SROCC 0.9054). NOT a ship candidate.

## CONTROL — ssim2 target + low weight ALSO fails (isolates the cause)

To test whether the cvvdp target shape was the culprit, reran with LARGE's
**`ssim2_gpu/100` target** (53,800 rows with non-null ssim2) at **train_w 0.2**
(manifest `v47_plus_large_ssim2_2026-05-30.toml`):

| | v47 | LARGE cvvdp-tgt tw0.6 | LARGE **ssim2-tgt tw0.2** |
|---|--:|--:|--:|
| JXL dial @best | 74.9 | 66.2 | **74.5** (≈v47 — NOT fixed) |
| CID22 SROCC | 0.866 | 0.732 | **0.699** |
| AIC-3 SROCC | 0.768 | 0.638 | **0.595** |
| AIC-4 SROCC | 0.885 | 0.683 | **0.484** |
| TID SROCC | 0.793 | 0.807 | 0.784 |
| KonJND SROCC | 0.419 | 0.435 | 0.477 |

**The ssim2 target did NOT recover held-out — it cratered just as hard (worse
on AIC-4: 0.484).** This isolates the cause: it is **NOT the target shape**,
it is **LARGE's near-lossless-only distribution** (iwssim ≥ 0.935). Adding
~50–70k near-lossless rows — any target, any weight — pulls the metric's
calibration toward the top and destroys the low/mid-q discrimination
CID22/AIC-3/AIC-4 depend on. And neither variant lifts the JXL dial.

**"Add the existing LARGE JXL data" is FALSIFIED on both target variants.**
The data is unusable as-is — it lacks the low-q JXL.

## Two redirections (both still open)

1. **Full-range JXL is mandatory, not optional.** The fix needs JXL spanning
   q5→q100 (balanced), with ssim2 target. The low-q JXL (q5–q60) encodes are
   on disk but **unscored** — that is the genuinely-missing data. Without it,
   adding JXL skews calibration (proven twice here).
2. **A spline-anchor refit (task #42) will NOT fix it.** The dial spline is a
   *monotone 1D function of raw net output* — it is rank-preserving. JXL @best
   reads 74.5 < jpeg's 92.6 *only because the net's raw output ranks
   near-lossless JXL below jpeg q100* (feature L2(jxl@d0.025, jpeg@q100)=4.85,
   ~24× a real jpeg q-step). A spline refit can globally rescale the
   near-lossless region (lifting ALL codecs, which would over-score jpeg q100)
   but **cannot selectively lift JXL** without contradicting jpeg at the same
   raw value. The underscoring is in the *net's learned ranking*, which only
   training changes.

### Conclusion: only full-range JXL training can fix it

Both LARGE variants fail (near-lossless skew craters held-out, never lifts the
dial); the spline angle can't selectively fix a codec. The single viable path:
**score the full-range on-disk JXL (q5→q100, balanced) with ssim2, retrain at a
moderate weight** — the balanced distribution avoids the skew-crater while
giving enough JXL signal to shift the net's ranking of near-lossless JXL toward
jpeg-q100-equivalent. The low-q JXL (q5–q60), unscored on disk, is the
load-bearing missing data. (Risk acknowledged: even balanced full-range JXL may
not fully close it if the OOD features are intrinsically hard — but it is the
only mechanism that *can*, and it's a clean experiment.)

Control bake: `v47_plus_large_ssim2_2026-05-30.bin`. NOT a ship candidate.
