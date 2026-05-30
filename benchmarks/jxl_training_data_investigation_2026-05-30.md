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

<!-- RESULT (pending): jxl @best dial, per-codec dial, held-out rank deltas vs v47 -->
