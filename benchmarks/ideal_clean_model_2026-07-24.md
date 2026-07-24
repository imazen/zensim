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
| **`signed_cbrt`** ★ | **0.80** | **0.77** | **0.67** | **0.79** | **63%** | **1.000** | **~0.57** | **passes every gate — the clean ideal** |
| winsor-guard | 0.80 | 0.89 | 0.90 | 0.89 | 92% | 0.977 | **−0.23** | best FR+corruption; **diffmap BROKEN** |
| ridge (unconstrained) | 0.80 | 0.57 | 0.59 | 0.84 | 3% | **0.82** | **−0.13** | no sign mask → dial + diffmap both broken |

- **`signed_cbrt` is the only row clean on all four columns.** Raw wins M3 but fails
  corruption; winsor wins FR+corruption but the diffmap sign-flips; ridge breaks both
  dial and diffmap (it's the sign constraint — BVLS — that makes them clean).
- The MLP `winner_dial` (CID22 0.894) is off this table on purpose: its dial fails
  (0.922) and its M3 ≈ 0. Higher rank, but not a *clean* model — it can't steer.

## Honest status

- **Clean on the meaningful gates:** dial monotonicity 1.000, diffmap M3 ~0.57
  (coherent — vs the MLPs' ~0), corruption 63%, foldable-mass 100%.
- **G1 dial *range* (p5≤25 ∧ p95≥85)** needs a full-0-100 anchor at 720 — the
  `multiband_anchor_dial100` is 372-feature; splining on `cid201` (human_score∈[0.4,0.9])
  gives a monotone but narrow range. Mechanical fix: extract v2 on the ~2000 dial-anchor
  images → a 720 anchor, then `bake_dial_refit add-spline`.
- **Rank ceiling.** CID22 0.80 / CSIQ 0.77 is the price of clean: additive +
  sign-constrained + foldable-only + smooth-shaped. The MLP's 0.89 buys ~+0.09 CID22 at
  the cost of the dial and the diffmap — not a trade the closed-loop product should make.
- **Next levers to push the clean rank** (without breaking cleanliness): a smooth
  *saturating* runtime transform (tanh/soft-sign — bounded derivative everywhere → keeps
  M3 like raw AND compresses like winsor → could lift FR-rank + corruption together;
  needs a new `FeatureTransform` + its block-pool test); CID22-recovering target mix
  (IW-SSIM/cvvdp-mix per `[[feedback_cvvdp_scalar_target_dead_end]]` — a MIX, not scalar);
  soft-peak fold (3 more foldable families, task #48 follow-on).

Recipe/corpus: `ext720-foldable-2026-07-24`, `screen_720_smooth.tsv`; tool changes:
`linear_projections` `--ridge` + robust `_chol` (NaN-guard + wider jitter for masked cols).
