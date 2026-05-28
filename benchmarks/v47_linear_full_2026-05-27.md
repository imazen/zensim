# v47-linear — full evaluation vs shipped v47 (codec / corruption / near-lossless)

User (2026-05-27): "0-hidden MLP, nearly just a weights table, more robust to
catch image-engine errors than compression, good at corruption and
near-lossless too." Built via the existing trainer with effectively-linear
config (`hidden=8` + `leaky_alpha=1.0` makes the chain a true linear
projection); `monotone_cbc + monotone_strict` for masked-monotone via the
same 300/372 sign mask v47 uses; `auto_transforms` **disabled** (probed and
found they HURT linear capacity — KADID 0.682 → 0.825 when off — they were
tuned for the MLP's representation, not a pure linear projection).

Bake: `/mnt/v/output/zensim/bakes/v47_linear_h8_strict_noauto.bin` (15.8 KB,
roughly half v47's 27 KB; ~3 k params nominal but most near-zero — the
linear projection is what matters).

## Held-out Mohammadi panel (linear vs shipped v47)

| corpus | linear SROCC | v47 SROCC | Δ | linear Z-RMSE | v47 Z-RMSE |
|---|--:|--:|--:|--:|--:|
| CID22 | 0.6324 | **0.8657** | −0.23 | 0.773 | 0.512 |
| **KADID** | **0.8251** | 0.7933 | **+0.03** | **0.569** | 0.613 |
| **TID** | **0.8150** | 0.7927 | **+0.02** | **0.571** | 0.577 |
| KonJND | 0.3221 | 0.4185 | −0.10 | 0.960 | 0.932 |
| AIC-3 | 0.5914 | 0.7680 | −0.18 | 0.805 | 0.620 |
| AIC-4 | 0.6280 | 0.8854 | −0.26 | 0.781 | 0.481 |

Surprise: **linear beats v47 on KADID and TID** (the two non-compression
analytic-distortion corpora) on both SROCC and Z-RMSE. Linear *loses* on
CID22 + AIC-3/4 (compression-distortion ranking — needs nonlinear capacity).

## Dial monotonicity (qsweep, 50 imgs × 19 q)

| bake | monotonicity | tied | violations |
|---|--:|--:|--:|
| v47 | 94.33% | 0.33% | 51 |
| **v47-linear** | **97.11%** | 0.78% | **26** |

**Linear has half v47's strict-monotonicity violations.** Cleaner dial — a
real win for codec-target binary search.

Near-lossless tail (q95 medians) — the part the user hoped linear would
extend:

| q | v47 median | v47-linear median |
|---|--:|--:|
| 85 | 82.12 | 66.52 |
| 90 | 85.83 | 68.78 |
| 95 | 88.50 | 70.26 |

**Linear's near-lossless tail is *compressed*, not extended.** v47 reaches
~88 at q95; linear plateaus at ~70. The hoped-for "more dial range
near-lossless" was *not* observed — the linear model's spline-calibrated
identity is ~75 (vs v47's 97.69), so the whole high-q region lives in
[55–75]. Falsified for this bake; would need an anchor reweighting to extend.

## Corruption-gate (gb82_dog photo, codec-corpus#7, 672 entries × {corruption, q20, q10})

Global gate (corruption-score < honest-q20-score) — overall + per-region:

| metric | v47 | v47-linear | Δ |
|---|--:|--:|--:|
| **overall vs q20** | 19.6% | **22.3%** | +2.7 pp |
| overall vs q10 (stricter) | 10.7% | 13.1% | +2.4 pp |
| whole-image | 37.1% | **41.7%** | +4.6 |
| frac2 | 37.0% | 38.9% | +1.9 |
| frac4 | 26.9% | 28.7% | +1.8 |
| sq64 | 13.0% | 14.8% | +1.8 |
| **sq16** | 0.0% | **5.6%** | first non-zero! |
| sq8 | 0.0% | 0.0% | (global metric limit — needs #33's local head) |

**Linear catches more corruption at every granularity** (and reaches sq16,
which v47 zeros). Best per-family wins:

| family | v47 | v47-linear | Δ |
|---|--:|--:|--:|
| channel_zero_r | 5.6% | 27.8% | **+22.2 pp** |
| channel_swap_rb | 0.0% | 16.7% | +16.7 |
| edge_border_all_k2 | 33.3% | 66.7% | +33.3 |
| edge_border_top_k4 | 0.0% | 33.3% | +33.3 |
| composite_premul_as_straight | 44.4% | 55.6% | +11.1 |
| composite_wrong_bg_white | 33.3% | 44.4% | +11.1 |
| channel_max_r | 33.3% | 44.4% | +11.1 |
| block_copy_wrong | 38.9% | 44.4% | +5.5 |
| block_garbage | 55.6% | 61.1% | +5.5 |

Linear gives up small ground on `block_gray` (−5.6) and `channel_zero_g`
(−11.1), but wins more than it loses. The improvement is consistent across
**image-engine-style errors** (channel swaps/zeros, edge defects, composites)
— the user's exact target.

## Honest verdict

The user's intuition was **partly right**:

| claim | result |
|---|---|
| "more robust to catch image-engine errors than compression" | ✓ **CONFIRMED** — +2.7pp overall corruption gate, +5–33pp on channel/edge/composite families |
| "good at corruption" | ✓ confirmed (above) + the only bake with non-zero sq16 detection |
| "good at near-lossless" | ✗ **FALSIFIED** for this bake — near-lossless dial is *compressed* to ~70, not extended. Spline-calibration choice; a follow-up could anchor identity → 97 like v47 does |
| (implicit) "competitive on codec rank" | partial — *better* than v47 on KADID/TID; *worse* on CID22/AIC by 0.18–0.26 SROCC |

## Ship decision (USER-GATED — this is a new public API)

Shipping requires adding `ZensimProfile::A_Linear` (a new public enum
variant). The case:

- **For:** real dial-monotonicity improvement, broader corruption coverage,
  better on KADID/TID, smaller bake (15.8 KB), interpretable weights.
- **Against:** −0.23 CID22 SROCC, near-lossless compression, codec-rank loss
  on the compression corpora (AIC-3/AIC-4 down ~0.18–0.26).

**Recommendation:** ship as `ZensimProfile::A_Linear` *if* the corruption
+ dial-monotonicity win is the primary use case (regression-test gating /
image-engine-error detection). For codec quality ranking, keep `Profile::A`
(v47) — the linear variant is not a replacement, it's a **complementary
sibling tuned for robustness over codec-rank fidelity**.

Follow-up worth doing before any ship:
1. **Spline anchor re-weighting** to extend the near-lossless dial range
   (target identity → 95 like v47, not 75) — straight-forward post-train fix.
2. **Trainer "soft-monotone-keep-72" mode** to recover the MVP-Python's
   CID22 ~0.82 (the trainer's strict mode drops 72 free features; MVP
   showed those 72 are worth ~0.16 SROCC).
3. **#40 fix** (hidden=1 bake-emit bug) — would let the bake be even
   smaller / more genuinely linear.
