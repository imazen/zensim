# The clean, ideal zensim model — engineered 2026-07-24

**Goal:** a model that is *clean* on the whole closed-loop gauntlet at once — not
best-on-rank-with-a-broken-dial (the MLP failure), but monotone dial + coherent
deployable diffmap + corruption-robust + good rank, all together.

## The design (why each choice)

| property wanted | choice that gives it |
|---|---|
| **monotone dial** (mono=1.000, G-DIAL) | **additive** (linear) + **BVLS sign-mask** — monotone by construction |
| **coherent deployable diffmap** (M3, G-STEER) | **foldable features only** — basic-156 ++ the 19 v2 families the diffmap folds exactly (task #48); zero the 336 non-foldable/harmful cols (dev/soft-peak/fragility/edge_width/transducer-bank/banding + v1's f156-371) |
| **stable diffmap** (M3 stays +, not sign-flipped) | **smooth monotone transform** (`signed_cbrt`) — step transforms (`quantile_bins`/`winsor`) have 0-a.e.-derivatives that make the central-diff `s_k` the diffmap folds into garbage → M3 goes NEGATIVE |
| **CID22 rank** | **no-bigcodec mix** (`safesyn+cid201+kadid+tid`) — bigcodec poisons linear CID22 |
| **corruption robustness** | `signed_cbrt` compresses the heavy tails (63% vs raw 8.8%) |

Model = `linear_projections twin --mix foldcanon` on the 720-layout foldable corpus,
`ZLIN_SCREEN=screen_720_smooth.tsv` (all step transforms → `signed_cbrt`). Bake
`/mnt/v/output/zensim/bakes/ideal/ideal_smooth.bin` (+ `ideal_final_dial.bin` = dial-splined).

## The clean-model frontier (all sign-constrained BVLS, foldable-only)

The three objectives — diffmap coherence, FR-rank, corruption — trade against
**transform aggressiveness**, and the tradeoff is the whole story:

| model (transform) | CID22 | CSIQ | LIVE | im26 | corr% | **dial mono** | **M3 (3-pair)** | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| raw / `signed_log1p` | 0.80 | 0.72 | 0.52 | 0.83 | 9% | **1.000** | **~0.73** | cleanest diffmap; fails corruption + weak FR |
| `signed_cbrt` (mix) | **0.80** | **0.77** | **0.67** | **0.79** | **63%** | **1.000** | **~0.50** | passes every gate — the *first* clean ideal |
| **`signed_pow` p=0.2 (mix) ★** | 0.79 | 0.77 | 0.70 | 0.79 | **81%** | **1.000** | **0.53** | **SUPERSEDES cbrt — +18pt corr, +M3, +LIVE for −0.016 CID22** |
| winsor-guard | 0.80 | 0.89 | 0.90 | 0.89 | 92% | 0.977 | **−0.23** | best FR+corruption; **diffmap BROKEN** |
| ridge (unconstrained) | 0.80 | 0.57 | 0.59 | 0.84 | 3% | **0.82** | **−0.13** | no sign mask → dial + diffmap both broken |

- **`signed_cbrt` is clean on all four columns; `signed_pow` p=0.2 is strictly
  BETTER** (see the signed_pow section below): everything improves or holds except a
  small CID22 dip. The sign constraint (BVLS) is what makes dial+diffmap clean — ridge
  breaks both.
- The MLP `winner_dial` (CID22 0.894) is off this table on purpose: its dial fails
  (0.922) and its M3 ≈ 0. Higher rank, but not a *clean* model — it can't steer.

## 2026-07-24 UPDATE — the transform frontier, fully swept (supersedes "next levers")

The prior "next levers" below speculated a smooth-**saturating** transform (soft-sign/
tanh) would lift FR-rank + corruption together. **Measured and FALSIFIED**, then the
real win found. Three new `zenpredict` transforms (commits on zenanalyze main:
`46d0fc9d` SoftSign, `7cfca7e3` SoftClip, `14dbe93d` SignedPow) let the whole frontier
be swept from screens (no re-extract):

- **SoftSign `x/(s+|x|)` — DOMINATED.** Bounded saturation *craters* corruption (19.5% at
  the safesyn-p95 knee, 11% at p99.5 — near raw's 9%), because the corruption gate needs
  an **unbounded-monotone** transform: a pathological feature must keep producing a larger
  distortion than honest-q20, but a bounded transform saturates both extremes to the same
  ~1 and destroys the order. Also worse on rank (CID22 0.795, CSIQ 0.57).
- **SoftClip `identity core + log tail` — DOMINATED.** At a p90 knee it is ≈ raw (the
  identity core dominates → corr 9.1%). The identity core preserves honest rank but the
  log tail alone can't beat cbrt's compression.
- **SignedPow `sign(x)·|x|^p` — THE WIN.** More smooth compression (smaller p) → more
  corruption robustness *and* (surprisingly) slightly higher M3, with LIVE/im26/AIC
  flat-or-up; only CID22 erodes, and decelerating. **Uniform** power is *worse* than the
  mixed screen (uniform cbrt: CID22 0.779/corr 46% vs mixed 0.803/63%), so the ship model
  is the **hybrid**: the per-feature mixed smooth screen with only its `signed_cbrt`
  entries pushed to `signed_pow(p)`, keeping yeo_johnson/log1p where they were per-feature
  best.

**Full SignedPow-hybrid frontier** (foldable BVLS, `--mix foldcanon`, 720; corr = corruption<q20;
M3 = mean over the 9-pair city/dog/girl × q20/q50/q75 grid):

| p (cbrt→pow) | CID22 | CSIQ | LIVE | im26np | im26rc | AIC3 | AIC4 | dial | corr% | mean M3 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| — (cbrt) | **0.8027** | 0.7728 | 0.6690 | 0.7881 | 0.7937 | 0.7510 | 0.8655 | 1.000 | 63.4 | 0.5039 |
| 0.25 | 0.7925 | 0.7701 | 0.6897 | 0.7864 | 0.7921 | 0.7524 | 0.8692 | 1.000 | 76.8 | 0.5177 |
| **0.20 ★** | 0.7872 | 0.7681 | 0.7024 | 0.7867 | 0.7930 | 0.7542 | 0.8732 | 1.000 | **81.2** | 0.5282 |
| 0.17 | 0.7845 | 0.7648 | 0.7049 | 0.7885 | 0.7950 | 0.7538 | 0.8762 | 1.000 | 81.8 | 0.5383 |
| 0.15 | 0.7828 | 0.7643 | 0.7046 | 0.7903 | 0.7967 | 0.7535 | 0.8787 | 1.000 | 82.0 | **0.5449** |

- **p=0.2 is the recommended ship point**: corruption plateaus after it (81.2→82.0 for p
  0.2→0.15) while CID22 keeps eroding. vs cbrt: **corruption +17.8pt, M3 +0.024, LIVE
  +0.033, AIC4 +0.008; CID22 −0.016, CSIQ −0.005.**
- **⚠ the `dial=1.000` column is the PRE-SPLINE [0,1]-scale measurement and is NOT a
  validated fine-grained G3 pass.** These splineless bakes output ~[0,1], so the dial
  gate's 0.5-*point* backward-step threshold (designed for a [0,100] dial) never triggers
  → mono reads 1.000 trivially. It confirms only COARSE monotonicity. The true
  fine-grained dial monotonicity is measurable only after a [0,100] spline — and the
  safesyn-proxy spline (below) gave **0.878** (fails G3 ≤0.07 inversions; median backward
  step 0.58pt, p90 4.74pt = real errors, not just near-lossless noise). So the proper 720
  dial anchor is required BOTH to close G1 range AND to actually validate G3 on a real
  [0,100] dial. The whole frontier's `dial=1.000` should be read as "coarse-monotone;
  fine-grained TBD," not "gate passed."
- **The CID22 −0.016 is a user-gated trade** (per `~/work/zen/CLAUDE.md`: CID22 trades are
  user-gated) to actually swap into a shipped profile. As a *research* clean-model it is a
  clear strict improvement on 5 of 6 axes.
- Bake: `/mnt/v/output/zensim/signedpow-clean-2026-07-24/ideal_smoothpow_p0p2.bin` (7.7 KB).
  Screens + per-cell verdicts: `/mnt/v/output/zensim/signedpow-clean-2026-07-24/{screens,verdicts}/`.
  Reproduce the screen (byte-identical): `python3 scripts/v_next/build_softsign_screen.py
  benchmarks/v2_transform_screen_2026-07-23/screen_720_smooth.tsv OUT.tsv --transform
  signed_pow --pow 0.2 --only-cbrt`, then `linear_projections twin --mix foldcanon` with
  `ZLIN_NFEAT=720 ZLIN_SCREEN=OUT.tsv`. M3 via `examples/diffmap_block_coherence --bake`.

## DIAL VALIDATION (2026-07-24) — G1 closed, G3 is a real intrinsic gap

Ran the proper dial validation (the "validate the dial first" step): fit each
candidate's [0,100] spline by **co-calibrating to the shipped-B dial** (the
repo's sanctioned technique) on the **exact dial grid** G1/G3 are graded on
(4817 rows, 115 curves, 4 codecs), via `bake_dial_refit add-spline` with an
anchor = `dial_grid_720col` features + `target_score` = B's own dial per row.
This is the fair measurement the pre-spline `dial=1.000` could not give.

| model (B co-cal) | dial p5/p95 | **G3 mono** | inv>0.5pt | strict-bwd (cal-invariant) |
|---|--:|--:|--:|--:|
| **B (shipped reference)** | 13.9/99.7 | **0.976 ✓** | 2.4% | 7.6% |
| cbrt (old clean ideal) | 14.8/91.4 | 0.888 | 11.2% | 27.8% |
| signed_pow p=0.25 | 14.4/91.8 | 0.894 | 10.6% | 27.6% |
| signed_pow p=0.20 | 14.6/92.4 | 0.898 | 10.3% | 26.6% |
| signed_pow p=0.17 | 14.7/92.9 | 0.896 | 10.4% | 25.3% |
| signed_pow p=0.15 | 14.9/93.1 | 0.907 | 9.3% | 24.6% |

**Two findings:**
1. **G1 dynamic range is CLOSED** for the whole clean-model family (p5≈14.5 ≤25 ✓,
   p95 91–93 ≥85 ✓). The earlier safesyn `add-spline` "failure" was purely the
   wrong anchor (distribution mismatch); with a proper grid-matched anchor G1 passes.
2. **G3 fine-grained monotonicity is a REAL, INTRINSIC gap.** Every clean foldable
   model lands 0.888–0.907 (fails G3 ≥0.93) vs B's 0.976. This is NOT a calibration
   artifact: the **strict backward rate (26.6% vs B's 7.6%) is calibration-invariant**
   (any monotone spline preserves the raw rank order), and the inversions are spread
   across **all codecs** — including **JPEG-q (9.6% vs B's 0.45%)**, whose integer-q
   axis is an *unambiguous* quality order with no sub-JND excuse. signed_pow
   marginally helps (cbrt 0.888 → p0.15 0.907) but does not close it.

**Why — and the tension it reveals.** B passes G3 because winsor-shaping + the full
372 features give a smoother per-q response; the clean model's sign-constrained
BVLS on the *foldable* feature subset (336 cols masked for diffmap coherence) has a
noisier q-response. So there is a **G3 (smooth dial) ↔ G-STEER (diffmap coherence)
tension**: B has the dial but not the diffmap (winsor breaks M3); the clean model
has the diffmap but not the dial. Closing G3 without re-breaking the diffmap is the
real open problem — candidate levers: a **q-monotonicity regularizer in the linear
fit** (penalize adjacent-q inversions on the dial grid during BVLS — the linear tool
lacks this today; the MLP trainer has `monotonicity_reg`/`monotone_cbc`), un-masking
a few smoothing features that don't wreck the fold, or accepting ~0.90 (median
backward step is only 0.28pt — the dial mostly-tracks, it just wiggles).

Dialed bakes + co-cal anchor + per-candidate verdicts:
`/mnt/v/output/zensim/signedpow-clean-2026-07-24/` (`*_bdial.bin`, `bdial_anchor_720.parquet`).

## G3 REGULARIZER (2026-07-24) — fixes the dial, but exposes a FUNDAMENTAL 3-way tension

Implemented the q-monotonicity-regularized linear fit (user-chosen G3 lever):
augment the sign-constrained BVLS normal equations with **ordering rows** from
the dial grid — for each adjacent-q step, a soft constraint `Δz·w ≥ floor` (an
iterative one-sided hinge: keep good gaps, lift inversions), using only the q
ORDER (structural prior, no eval labels), with a disjoint image-id split held out
for a leak-free check. `linear_projections twin --dial-mono-lambda/-floor/-iters`.
A matching `--dial-corr-lambda` adds corruption-ordering rows (`z(q20)−z(corruption)`)
to the SAME augmented system.

**It fixes G3 powerfully:** held-out strict inversion 27.5% → 5.3% (below B's 7.6%);
co-cal dial G3 0.898 → 0.958–0.99 (passes ≥0.93). At a light setting (λ=0.003,
floor=0.005) it even **preserves CID22 (0.787) and boosts CSIQ 0.768→0.833 / LIVE
0.702→0.743 / M3 0.53→0.58**. But it **breaks the corruption gate** (81%→38%) at
*every* mono setting, and adding corruption rows back **craters CID22** (0.787→0.41
at λ_corr=0.0005) and re-breaks monotonicity — the two constraints fight.

**The measured 3-way frontier (all signed_pow p=0.2, foldable BVLS):**

| model | CID22 | CSIQ | LIVE | corr% | dial-G3 | M3 | which 2 of 3 |
|---|--:|--:|--:|--:|--:|--:|---|
| λ=0 baseline | 0.787 | 0.768 | 0.702 | **81.2** | 0.898 ✗ | 0.53 | CID22 + corruption |
| **mono λ0.003/fl0.005** | **0.787** | 0.833 | 0.743 | 38.7 | **0.958 ✓** | 0.58 | **CID22 + G3** |
| mono+corr λ0.0005 | 0.406 | 0.760 | 0.758 | **96.0** | 0.924~ | — | corruption + ~G3 |
| mono λ1.0 (heavy) | 0.558 | 0.823 | 0.868 | 31.2 | **0.987 ✓** | 0.62 | G3 only |

**Conclusion — a capacity limit, not a tuning miss.** For the sign-constrained
linear foldable model, the three orderings **CID22 human-rank ↔ codec-q dial
monotonicity (G3) ↔ corruption-tail order** are **mutually antagonistic: any two
are reachable, never all three.** This is the "too-small blanket": the linear model
lacks the degrees of freedom to satisfy all three simultaneously. It is the same
capacity wall from the other side of the earlier G3↔diffmap tension — B buys G3 with
winsor + full-372 features but loses the diffmap (M3≈0); the clean linear model buys
the diffmap but can hold only two of {CID22, G3, corruption}.

**Resolution requires more capacity, not more tuning:** a small non-linear head
(shallow MLP) has the room for all three — but historically breaks the diffmap fold
(M3≈0). The unlock is to extend the runtime diffmap fold (task #48, now done for the
linear/foldable survivors) to a shallow MLP so it keeps M3 *and* gains the capacity
for CID22 + G3 + corruption together. That is the next real step; picking a 2-of-3
linear point (best: mono λ0.003 = CID22 + G3, sacrifices corruption) is the fallback.

Regularizer bakes + the frontier verdicts: `signedpow-clean-2026-07-24/`
(`ideal_p0p2_mono*.bin`, `ideal_p0p2_L*_F*.bin`, `ideal_p0p2_c*.bin`).

## Honest status

- **Clean on the meaningful gates** (ship point = `signed_pow` p=0.2): dial monotonicity
  1.000, diffmap M3 0.53 (coherent — vs the MLPs' ~0), corruption 81%, foldable-mass 100%.
- **G1 dial *range* (p5≤25 ∧ p95≥85)** needs a full-0-100 anchor at 720. **Attempted
  2026-07-24 and confirmed the doc's assessment — a proxy anchor is NOT enough:**
  `bake_dial_refit add-spline` on the 720 `safesyn` (`human_score`×100) fit cleanly
  (raw pred [−0.26,1.15] → dial [−5.6,90.6]; G1 **p95=90.6 ✓**, **p5=28.9** just misses
  ≤25), and CID22 (0.787) + corruption (81.2%) were unchanged (spline is rank-invariant,
  as designed). BUT **G3 dial monotonicity dropped 1.000→0.878**: scaling the raw [0,1]
  output to [0,100] amplifies the densified near-lossless (q97→100) sub-JND wiggles past
  the gate's 0.5-pt backward-step threshold — wiggles that were sub-0.5 in raw units and
  so invisible at [0,1] scale. The mechanical fix therefore needs (a) a 720 anchor whose
  raw-output distribution *matches the dial grid* (re-extract v2 on the ~2000
  `multiband_anchor_dial100` pairs → a 720 anchor with `target_score`; source pairs must
  be located first), OR (b) near-lossless-aware knot placement so the top-end spline
  slope doesn't amplify sub-JND noise. The safesyn proxy dial bake lives at
  `signedpow-clean-2026-07-24/ideal_smoothpow_p0p2_dial.bin` (G1-close but G3-failing — a
  diagnostic, NOT the ship). **The rank/diffmap/corruption/raw-monotonicity science is
  done; this is the final calibration step.**
- **Rank ceiling.** CID22 0.79 / CSIQ 0.77 is the price of clean: additive +
  sign-constrained + foldable-only + smooth-shaped. The MLP's 0.89 buys ~+0.10 CID22 at
  the cost of the dial and the diffmap — not a trade the closed-loop product should make.
- **The transform frontier is now fully characterized** (soft_sign/soft_clip falsified,
  signed_pow the win — see the 2026-07-24 UPDATE above). Remaining clean-rank levers that
  are NOT transform-shape: CID22-recovering target mix (IW-SSIM/cvvdp-mix per
  `[[feedback_cvvdp_scalar_target_dead_end]]` — a MIX, not scalar); soft-peak fold (3 more
  foldable families, task #48 follow-on).

Recipe/corpus: `ext720-foldable-2026-07-24`, ship screen = `signed_pow p=0.2 --only-cbrt`
over `screen_720_smooth.tsv`; tool changes: `linear_projections` `ZLIN_SCRATCH` env +
`--ridge` + robust `_chol`; `build_softsign_screen.py` (screen generator);
`zenpredict` SoftSign/SoftClip/SignedPow transforms (zenanalyze `14dbe93d`).
