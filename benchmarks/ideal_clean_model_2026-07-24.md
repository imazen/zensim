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
  +0.033, AIC4 +0.008; CID22 −0.016, CSIQ −0.005.** dial monotonicity stays a perfect 1.000.
- **The CID22 −0.016 is a user-gated trade** (per `~/work/zen/CLAUDE.md`: CID22 trades are
  user-gated) to actually swap into a shipped profile. As a *research* clean-model it is a
  clear strict improvement on 5 of 6 axes.
- Bake: `/mnt/v/output/zensim/signedpow-clean-2026-07-24/ideal_smoothpow_p0p2.bin` (7.7 KB).
  Screens + per-cell verdicts: `/mnt/v/output/zensim/signedpow-clean-2026-07-24/{screens,verdicts}/`.
  Reproduce the screen (byte-identical): `python3 scripts/v_next/build_softsign_screen.py
  benchmarks/v2_transform_screen_2026-07-23/screen_720_smooth.tsv OUT.tsv --transform
  signed_pow --pow 0.2 --only-cbrt`, then `linear_projections twin --mix foldcanon` with
  `ZLIN_NFEAT=720 ZLIN_SCREEN=OUT.tsv`. M3 via `examples/diffmap_block_coherence --bake`.

## Honest status

- **Clean on the meaningful gates** (ship point = `signed_pow` p=0.2): dial monotonicity
  1.000, diffmap M3 0.53 (coherent — vs the MLPs' ~0), corruption 81%, foldable-mass 100%.
- **G1 dial *range* (p5≤25 ∧ p95≥85)** needs a full-0-100 anchor at 720 — the
  `multiband_anchor_dial100` is 372-feature; splining on `cid201` (human_score∈[0.4,0.9])
  gives a monotone but narrow range. Mechanical fix: extract v2 on the ~2000 dial-anchor
  images → a 720 anchor, then `bake_dial_refit add-spline`. **This is the one remaining
  step to a fully-shippable dial; the rank/diffmap/corruption science is done.**
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
