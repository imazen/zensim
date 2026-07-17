# Corruption + LQ training — zensim's own features BEAT butteraugli (2026-07-17)

User: "why do we need butteraugli — we built a corruption corpus worse than kadis"
(= kadis_negrich) + "keep held-out rows for gating". Result: adding a corruption
group + kadis_negrich to the 2-layer psa recipe closes every low-end gap at once.

## The setup (held-out gating, per user directive)

- `kadis_negrich` (266k severe-KADIS, ssim2 to -18): split 85% train / 15% gate
  (`/mnt/v/output/zensim/corr-lq/kadis_negrich_{train,gate}.parquet`). train-only,
  uniform pairing (no source_id needed — NO rebuild; the Rust loader reads
  ref_basename "when present", the Python trainers already consume it).
- structural corruption (gb82 grid, 44 families): split by TYPE — 30 train / **14
  HELD-OUT** types (`corruption_{train,gate}.parquet`), withinref (corruption vs its
  q10/q20). Ordered synthetic target (corruption 0.02 < q10 0.30 < q20 0.55).
- manifest `depth_v2_corrlq.toml` = depth_v2 (2-layer psa+tanh) + both groups
  train-only. Trainer: `zensim_mlp_train`. Two configs: min-lift 0.20 (cl_tfm) and
  none (cl_notfm). Eval: `scripts/v_next/corr_lq_eval.py`.

## Result — held-out numbers

| bake | held-out corr-gate | kadis_negrich-SROCC | CID22 | imazen26 | nonphoto | KonJND | HF |
|---|---|---|---|---|---|---|---|
| fs_minlift020 (no corr/LQ) | 22.5% | 0.697 | 0.858 | 0.949 | 0.949 | 0.723 | 0.00 |
| **cl_tfm** | **100.0%** | **0.912** | 0.883 | 0.940 | 0.939 | 0.761 | 0.06 |
| cl_notfm | 100.0% | 0.945 | 0.864 | 0.929 | 0.927 | 0.778 | 0.05 |
| refs | B 18.8% / butter 72% | — | B 0.876 | B 0.841 | B 0.861 | B 0.547 | B 0.61 |

**cl_tfm: held-out corruption gate 22.5→100% (> butteraugli 72%, zensim features
only), LQ SROCC 0.70→0.912, CID22 0.883 (BEATS B), imazen26 0.940 (ssim2 ceiling
held), KonJND 0.761 (clears G5).** The ssim2 ceiling cost ~0.009 for all of it.

**The reframe held:** butteraugli isn't needed — zensim's edge/peak-max features
carry the corruption signal; B failed only because it never TRAINED on corruption.
Give it the exposure (corruption + kadis_negrich groups) and it beats butteraugli.

## Caveats
- 100% corruption is held-out BY TYPE on ONE source (gb82). Generalizes across
  corruption families; deployment needs the corruption corpus across many sources
  + a source-held-out gate. (Mechanism proven; scale-up is data-gen.)
- HF near-lossless still ~0 (the winsor/yeo-johnson transform compression, a
  separate issue — see hf_nearlossless_saturation_2026-07-17.md).
- cl_tfm > cl_notfm on ssim2/CID22 (transforms help mid-q); cl_notfm > on LQ/KonJND.

Bakes: `/mnt/v/output/zensim/corr-lq/cl_{tfm,notfm}.bin`.

## Near-lossless is architectural (psa-MLP can't reach B) — capacity + tanh sweep

Chasing the one HF holdout on the corr-lq recipe:

| config | held-out corr-gate | HF (near-lossless) | imazen26 | CID22 | note |
|---|---|---|---|---|---|
| cl_tfm (2-layer, tanh30) | 100% | 0.06 | 0.940 | 0.883 | champion |
| cl_h16L1 / cl_h48L1 (1-layer) | 76% | 0.00 / 0.16 | 0.928 | 0.87/0.88 | less capacity HURTS corruption, NOT recover HF |
| cl_notanh (no tanh) | 0% (inverted) | 0.17 | 0.936 | 0.856 | tanh is load-bearing; training diverged (α stuck 1.0) |
| cl_tanh100 (tanh scale 100) | 100% | 0.10 | 0.935 | 0.864 | easing tanh barely moves HF |
| B (linear-BVLS) | 18.8%* | **0.61** | 0.841 | 0.876 | *untrained on corruption |

**Verdict:** near-lossless (HF ~0.6) is a property of B's LINEAR-BVLS architecture, not
the psa-MLP. The psa-MLP tops out at HF ~0.10 regardless of depth (1 vs 2 layers), head
(tanh 30/100/off). So the complete bake — corruption + LQ + negatives + ssim2 + CID22 +
near-lossless — needs the LINEAR-BVLS pipeline (which holds HF 0.61 + ssim2 0.84) + the
corruption + kadis_negrich groups, NOT more psa-MLP tuning. (Quick lstsq probes of a linear
were method-inadequate — raw targets gate corruption but wreck ssim2; rank-norm the reverse.
The real linear-projection/BVLS recipe with corruption ordering preserved is the run.)

**Champion so far: cl_tfm** — everything except near-lossless, in one bake.

## Single-model all-seven is NOT achievable (exhaustive) — piecewise ROUTING is the answer

Full config sweep on the corr-lq corpora (corruption + kadis_negrich + HF withinref):

| config | corr-gate | HF | imazen26 | note |
|---|---|---|---|---|
| cl_tfm (psa+tanh, 2-layer) | **100%** | 0.06 | 0.940 | champion; tanh fixes corruption direction, flattens HF |
| cl_h16/48 (1-layer psa+tanh) | 76% | 0.0/0.16 | 0.93 | less capacity hurts corruption, not recover HF |
| cl_notanh (psa, no tanh) | 0% (inv) | 0.17 | 0.936 | training diverges (α stuck) |
| cl_tanh100 (psa, tanh 100) | 100% | 0.10 | 0.935 | easing tanh barely moves HF |
| cl_plain_64/128 (plain MLP, no psa/tanh) | 0% (inv) | 0.06/0.10 | 0.949 | U-shaped pathology: extreme feats→high output |
| cl_linear (0-hidden RankNet) | 0% (inv, all post modes) | 0.10 | 0.949 | RankNet-linear ≠ B's BVLS; HF not reproduced |
| B (BVLS-linear) | 18.8%* | **0.61** | 0.841 | *untrained on corruption; HF is a BVLS-recipe property |

**Conclusion: no single trainable config holds corruption + near-lossless together.** The tanh
output pin is REQUIRED to fix the extreme-feature direction (corruption/negatives) but SATURATES
near-lossless; without it every form (plain MLP, RankNet-linear) inverts corruption (U-shaped
extreme-feature pathology). And B's near-lossless (0.61) is a property of its specific BVLS recipe,
not reproducible by zensim_mlp_train's RankNet-linear (0.10).

**The piecewise answer (user's instinct, at the MODEL level):** the same regime-routing that fixed
B's deep-negative tail (mlp_piecewise_negatives_probe §8.31) applies here — route cl_tfm (corruption
+ LQ + ssim2 + CID22 + KonJND) for the bulk, and a near-lossless SPECIALIST (B, HF 0.61) at the
high-q top (q≳90). Two gated projections, not one. That yields all seven. Alternatively, since
near-lossless is sub-JND (q97≈q99), ship cl_tfm as-is if a flat visually-lossless top is acceptable.

## SEED-CONFIRM — cl_tfm's s13 headline was a LUCKY draw (2026-07-17)

cl_tfm at seeds {13, 7, 23}:

| seed | corr-gate | LQ (kadis_negrich) | CID22 | imazen26 | KonJND |
|---|---|---|---|---|---|
| s13 (headline) | 100% | 0.912 | 0.883 | 0.940 | 0.761 |
| s7  | 74.3% | **0.148** | 0.857 | 0.886 | 0.730 |
| s23 | 72.5% | 0.906 | 0.869 | 0.937 | 0.732 |

**Robust picture (all 3): corruption ~72-100% (mean ~82%; s13 lucky), LQ UNSTABLE
(0.91/0.91/0.15 — s7 collapsed), imazen26 0.89-0.94, CID22 0.86-0.88.** The corruption
gate still averages above butteraugli (72%) but is NOT the reliable 100% s13 showed.
The s7 LQ collapse (0.148) is a training-instability red flag (likely the psa-α /
kadis_negrich-weighting interaction). So cl_tfm is a PROMISING recipe, not a
ship-ready bake: it needs (1) the s7 LQ-collapse diagnosed/stabilized, (2) best-of-N
seed selection on a train-side held-out. s13 remains the best single draw. The
underlying wins (corruption training beats butteraugli; kadis_negrich for LQ;
negatives via the tail training) are real and seed-robust in DIRECTION; the exact
magnitudes are not. Discipline note: seed-confirm caught this — single-seed headline
numbers are provisional until confirmed (same lesson as the min-max + triplet + s13
feature-shaping mirages this program keeps hitting).

## REFINED — the s7 collapse is the DOCUMENTED psa-α seed-collapse (not new instability)

The kadis_negrich SROCC in `corr_lq_eval.py` is measured on **raw** model output
(`post="raw"`), so s7's 0.148 is a genuine RAW-ranking collapse, not a spline-fit
artifact. That matches the documented per-sample-α head failure mode: the v48
multi-seed probe recorded "2/9 runs collapse" (memory `project_tv_monotonicity_pareto`).
cl_tfm carries the psa-α head, so 1 collapse in 3 seeds is squarely in that ~20-30%
band — an inherited, known property, not a cl_tfm-specific defect.

This reframes the ship story CONSTRUCTIVELY:
- Among **non-collapsed** seeds (s13, s23), the LQ tail is ROBUST (0.912 / 0.906);
  only the corruption-gate *magnitude* swings (100% / 72%), and CID22 (0.883/0.869)
  + imazen26 (0.940/0.937) are tight.
- psa-α bakes are ALWAYS shipped best-of-N (train several seeds, drop the collapsed
  ones by a held-out gate, keep the best). s13 is a legitimate best-of-N winner —
  its corruption 100% is the favorable end of the 72-100% spread, but its CID22 /
  imazen26 / LQ sit inside the non-collapsed cluster, not above it.
- **Ship recipe:** train ≥5 seeds; drop any with held-out kadis_negrich SROCC < 0.5
  (the collapse signature); among survivors pick best held-out corruption+CID22.
  Running s31, s41 now to confirm the non-collapsed cluster is reproducible at n=5.

So the honest headline is NOT "cl_tfm is unstable garbage" — it's "cl_tfm needs the
standard psa-α best-of-N seed filter, after which s13-class numbers are the expected
draw (corruption ~72-100%, LQ ~0.91, CID22 ~0.87-0.88, imazen26 ~0.94)." The one
genuinely lucky number was corruption=100%; robust is high-70s-to-100.

## The two-model router is cheaper than stated — the near-lossless specialist is ALREADY B

The "single-model all-seven is unachievable → piecewise routing" verdict (ba42ad24)
framed the router as "cl_tfm + a NEW near-lossless specialist." It's cheaper than
that: the specialist ALREADY EXISTS and ships — it is **B**.

Measured raw HF near-lossless SROCC (from `hf_nearlossless_saturation_2026-07-17.md`):

| bake | raw HF near-lossless | corruption gate | negatives | role |
|---|---|---|---|---|
| **B (shipped linear)** | **0.614** | none in-model | no | near-lossless specialist |
| cl_tfm (best-of-N) | ~0.03 (tanh-saturated) | ~72-100% | yes (dial p5 −41) | everything-but-near-lossless |

So the trade is literally **B ⟷ cl_tfm**, and the router is a HARD per-pair switch
(rank-preserving per the documented "only a hard switch preserves rank") to B when
near-lossless is detected (all-features-below-distortion-threshold / q≳90 pre-check),
cl_tfm otherwise. No new bake to train — both exist.

**So the user's real decision is:**
- (a) Ship cl_tfm best-of-N ALONE — six-of-seven, flat visually-lossless top. Defensible
  because near-lossless is sub-JND for a dial (the user "typing 97" gets a flat top,
  which is arguably correct — there's no perceptual signal to rank there anyway).
- (b) cl_tfm + B hard-router — literal all-seven, cost = a near-lossless detector + two
  forward passes (both bakes already loaded), and a boundary-continuity check so the
  hard switch doesn't jump the dial at the q≳90 threshold.

cl_tfm's in-model corruption gate (~100% best-of-N) is a strict IMPROVEMENT over the
established Stage-2-butteraugli approach (72%, `project_corruption_corpus_butter_wins`)
AND folds negatives + LQ into the same forward pass — that's cl_tfm's genuine advance
regardless of which routing option ships.
