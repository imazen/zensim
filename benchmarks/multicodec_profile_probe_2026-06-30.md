# Multi-codec ssim2 profile probe — 2026-06-30

Can the `canonical-picker-2026-06-27` dataset (5.74M rows, 7 codec×modes,
372 zensim feat + score_ssim2, clean origin split) train a better Profile::A?
**Yes.** A bare probe already beats the shipped v47 on every codec/HF human-MOS
corpus and cracks the G5 KonJND wall.

## Probe setup
- Train: 1.2M rows stratified (300k each from zenjpeg/zenavif/zenjxl/zenwebp
  lossy `train.parquet`, origins {0,2,4,6,8}), ssim2-decile-stratified for low-q.
- Target: `score_ssim2` (--target-scale 1.0). Held-out val = origins {1,3,5}.
- `zensim_mlp_train --max-features 372 --hidden 128 --per-sample-alpha-head
  --mse-weight 0.6 --ranknet-weight 0.6 --epochs 60 --seed 17 --lr 1e-3
  --minibatch-size 256`. NO monotone-cbc, NO QAT (probe only).
- Held-out-origin ssim2-tracking SROCC = 0.9647.

## A/B vs shipped v47 (bake_verdict, identical harness + 372-col canonical val)

| Corpus | probe | A (v47) | Δ | kind |
|---|--:|--:|--:|---|
| CID22 | 0.8827 | 0.8657 | +0.017 | codec (gold) |
| AIC-3 | 0.7948 | 0.7680 | +0.027 | codec JND |
| AIC-4 | 0.8921 | 0.8854 | +0.007 | neural codec |
| KonJND | 0.5619 | 0.4185 | +0.143 | HF/near-lossless (G5 wall) |
| KADID | 0.6397 | 0.7933 | −0.154 | NON-codec analytic |
| TID | 0.7248 | 0.7927 | −0.068 | NON-codec analytic |

CID22 0.8827 within 0.007 of fast-ssim2 (0.8895). Probe wins all codec+HF
corpora; loses only the non-compression analytic guards (never trained on
blur/noise/geometric).

## Caveats
- Dial broken (G3 inversions 0.28) — no monotone-cbc. Not ship-ready as-is.
- ssim2-only target; 1.2M of 5.7M; 60 epochs. Full-scale + balanced should beat this.

## Next: balanced ship-candidate
multi-codec 5.7M + kadid/tid/konjnd train subsets (recover analytic guards) +
--monotone-cbc + QAT f16 (A's dial correctness) + 372. Scale on Hetzner.
Optional: score 5.7M variants with cvvdp+iwssim for A's proven mix target.

Bakes: /mnt/v/output/zensim-multicodec-probe/probe_372.bin (+verdict.md);
A baseline: A_baseline.verdict.md. Trainer ban-228 guard landed same day.

## Iteration 2: the monotone-vs-rank tension (2026-06-30)

Tried to make the probe ship-quality (recover analytic guards + monotone dial).
Two diagnostics on the same harness (bake_verdict, 372-col canonical val):

| recipe | CID22 | AIC-3 | AIC-4 | KonJND | KADID | TID |
|---|--:|--:|--:|--:|--:|--:|
| probe (multi-codec ssim2, no constraint) | **0.8827** | 0.7948 | 0.8921 | 0.5619 | 0.6397 | 0.7248 |
| + heavy analytic guards (k/t 0.3, kj 0.5) + mono-cbc | 0.7076 | 0.6265 | 0.6729 | 0.4372 | 0.8142 | 0.8168 |
| + mono-cbc-strict only (no guards) | 0.6795 | 0.6166 | 0.6505 | 0.4280 | 0.7627 | 0.7908 |
| shipped A (v47, mono-cbc + v47 recipe) | 0.8657 | 0.7680 | 0.8854 | 0.4185 | 0.7933 | 0.7927 |

**Findings:**
1. `--monotone-cbc --monotone-strict` on the pure-ssim2 multi-codec target
   **craters CID22** (0.88→0.68). Strict mode DROPS 72 features + sign-pins
   300; that constraint fits human-MOS-flavored targets (v47 used such a
   target and got 0.8657) but FIGHTS the ssim2-on-codec ranking. Notably it
   *raises* KADID/TID (the analytic distortions respect the sign-mask better).
2. Heavy analytic guards swamp the 1.2M codec rows (33k analytic rows at
   weight 1.1-combined ≈ 45% of training signal) → also craters codec corpora.
   Guard weights must be ≪ (≈0.03-0.05) so multi-codec dominates.
3. An output spline alone can't fix the probe's q-sweep dial inversions
   (monotone f(non-monotone) is still non-monotone).

**The tension:** multi-codec ssim2 is a strictly better RANK signal (CID22
0.88 > A 0.87) but A's strong correct-by-construction monotonicity caps CID22
with this data. The ship recipe must navigate this — candidates:
- (a) Max-rank + spline dial (V39-style, drop strict-CBC): best CID22, but
  not strong-feature-monotone (loses A's OOD-synthetic robustness).
- (b) Multi-codec as a GROUP inside v47's proven recipe (human_score-norm
  target + cid22_train + QAT), not as the sole pure-ssim2 target — may keep
  both. UNTESTED — most promising.
- (c) Soft monotonicity-reg (no strict, keep all 372 features) + spline.
- The cvvdp+iwssim target (score the 5.7M variants) is orthogonal and may
  help (a) and (b).

## Iteration 3 + correction: safesyn is NOT artificial distortions (2026-06-30)

User correction (verified against `coefficient/examples/generate_zensim_training.rs`
+ the synthetic-pipeline doc): **safesyn = synthetic/CID22-safe SOURCE tiles
(CID22-512, clic2025, kodak, corpus-builder JPEGs) run through real CODECS**
(mozjpeg/zenjpeg/zenwebp at quality levels), scored by ssim2/butteraugli. The
distortions are **codec compression, NOT artificial/analytic** (blur/noise/color/
geometric). "Safe" = CID22-leak-safe, not "artificially distorted".

Recipe (b) [multi-codec + safesyn + light guards + **monotone-cbc**]:
CID22 **0.6808** (A 0.8657) — same crater as mono-only. Adding more codec data
(safesyn) does NOT fix the monotone-vs-codec-ssim2 tension. KADID/TID rise
(+0.02) as before. **Confirmed: monotone-cbc-strict fundamentally caps CID22
on codec-distortion data; safesyn is not the safety ingredient.**

## Corrected ship recipe (the path forward)

The OOD-safety A gets from the strict monotone CONSTRAINT (which craters CID22)
should instead come from artificial-distortion **DATA**:
1. **Generate an artificial-distortion training set** — apply analytic
   distortions (the KADID taxonomy: gaussian/lens/motion blur, white/impulse/
   speckle noise, color shift/quantize/saturate, contrast, jpeg2k-style, spatial
   warp, sharpen) to the CID22-safe synthetic sources, score with ssim2.
   Leak-free (KADID/TID stay held-out guards). Does NOT exist yet — needs a new
   generation pass (Hetzner-able). [Open Q: fresh-generate vs KADID origin-split.]
2. **Train the ship bake**: multi-codec (rank, CID22 0.88) + artificial-distortion
   (OOD safety, learned not constrained) + light analytic guards, **NO strict-cbc**,
   + output PCHIP spline (dial). Hypothesis: high CID22 + OOD safety + working dial.
3. Validate: full panel + **tests/metric_invariants.rs** (blur>identity etc., the
   real OOD-safety gate) + dial monotonicity.
4. Scale on Hetzner (full 5.7M + cvvdp/iwssim mix target); HDR corpus gen in parallel.

## Iteration 4: artificial-distortion DATA works (TID-in-loop) + KADIS-700k (2026-06-30)

**Mechanism confirmed.** multi-codec (rank, 1.2M) + TID2013 (analytic, 3k, weight
1.0) in-loop, **NO strict-cbc**, validate held-out:

| corpus | tid-in-loop | probe (no tid) | A | note |
|---|--:|--:|--:|---|
| KADID (analytic, HELD-OUT) | **0.8662** | 0.6397 | 0.7933 | +0.23 transfer — TID taught analytic safety |
| CID22 (codec gold) | 0.8456 | 0.8827 | 0.8657 | small cost (tid weight 1.0 a bit high; tunable) |
| AIC-3 | 0.7883 | 0.7948 | 0.7680 | ≈ |
| AIC-4 | 0.8580 | 0.8921 | 0.8854 | ≈ |
| KonJND | 0.5072 | 0.5619 | 0.4185 | > A |

So artificial-distortion DATA (not the strict-cbc CONSTRAINT) delivers OOD safety:
held-out KADID 0.64→0.87 while CID22 only dips 0.88→0.85 (vs strict-cbc's crater
to 0.68). The TID weight is the knob (sweep down to recover CID22).

**KADIS-700k = the scaled version.** `/mnt/v/datasets/kadis700k/kadis700k.zip`
(42 GB): 140k pristine refs + `kadis700k_friqa.csv` (700k rows: dist_im,ref_im +
12 proxy FR-IQA scores [ssim,msssim,**iwssim**,mdsi,vsi,fsim,gmsd,sff,scqi,
add_gsim,sr_sim] — NOT SSIMULACRA2). Distorted images are NOT in the zip — the
dist_im name encodes type+level (`<ref>_<distortion>_<level>.png`), regenerated
from refs via the bundled distortion code. **Not integrated:** no extraction, no
generated distortions, no ssim2 score, no 372 features, not in any parquet.

### KADIS-700k integration plan (the scaled OOD-safety group)
1. Extract 140k refs. 2. Generate the 700k distortions (bundled code / our
   `gen_tid2013_distortions.py`, per the friqa.csv assignments). 3. Score ssim2
   (GPU) for target-consistency with the multi-codec group [friqa's iwssim is a
   faster alt]. 4. Extract 372 zensim features. 5. → parquet (f0..f371 +
   human_score=ssim2). Big GPU job → Hetzner.
6. Ship recipe: multi-codec (rank) + KADIS-700k (analytic safety, tuned weight) +
   PCHIP spline (dial), NO strict-cbc. Expect: CID22 ≥ A, KADID/TID ≥ A,
   metric_invariants (OOD) pass, monotone dial. Validate full panel + invariants.
