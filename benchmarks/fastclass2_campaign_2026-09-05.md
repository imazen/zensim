# FASTCLASS2 — a 156-or-156+cheap model with 944-class rank

Registration (frozen BEFORE any fit): **[`docs/PLAN_FASTCLASS2_2026-09-05.md`](../docs/PLAN_FASTCLASS2_2026-09-05.md)**.
Results append below. Nothing in the plan is edited after a number exists
except in a section explicitly labelled AMENDMENT.

## Status

| phase | what | state |
|---|---|---|
| pre-reg | plan + slice + owner levers | LANDED |
| §2 | identity localisation (measured, no fit) | **DONE — see plan §2** |
| G1 | control equivalence on this lane's build | pending |
| A | SET × WIDTH, 30 fits | pending |
| B | width extension, 6 fits | pending |
| C | head/depth, 9 fits | pending |
| D | id100 dial chain on the selected cell | pending |

## 0. What was already measured before the first fit

Two results needed no fit and are recorded here because they change how the
rest must be read.

**(a) The fast class's identity contamination is FOUR slots.** Plan §2, from
the D+free lane's 39-row 944-pools identity probe: `LUMA_MEAN_REF`
(f926/931/936/941) carries max |v| 0.688 and a 0.261 spread across references,
while every other slot in the 265 set stays under 4.8e-3 and all 33 other
raw-moment slots and all 24 class-C slots are identity-ZERO. New slice
`scripts/sota944/slice_basic156_free_nolumaref.txt` (261) drops exactly those
four; the producing walk is unchanged, so its W4 is identical to the 265 arm
by construction.

**(b) The gap to the 944 leaders is ONE axis, and the base recipe's cost on a
second axis is real.** Plan §1 and §4: at k = 3 the fast class already clears
the leaders' composite (0.8645 vs 0.8593) and CID22 (0.8863 vs 0.8848) bars
and misses only KonJND (0.4322 vs 0.4609, −0.029) — but the within-ref base
recipe pays −0.330 on `hfnlproxy` (0.4271 vs the control's 0.7572 and the
leaders' 0.70–0.74), an axis `product_composite` cannot see. Both are carried
as reported axes throughout.

---

## 1. GATE G1 — PASS, and it localises a pack-owner effect worth flagging

`F2_S265_H128_p_s4004` is the base recipe with every new lever unset, fit on
THIS lane's build. Against the incumbent `FC_D3_s4004` (wave-r4 pin,
2026-09-01), scored on the same root by the same binary:

| comparison | mismatching axes | composite |
|---|---|---|
| **RAW bakes** (trainer output, no pack) | **0 of 12** | `0.8634940859693885` on both, all 16 digits |
| PACKED bakes (`score_arm.sh`'s `bake_dial_refit pack`) | 4 of 12 | `0.8634920042634943` on both, all 16 digits |

**The trainer is bit-equivalent; the difference is entirely in the PACK.** The
packed deltas are csiq 7.2e-9, tid 1.0e-7, kadid 6.3e-7, **live 2.0e-5**, and
the packed files differ in SIZE (32,924 B vs 29,097 B), so different weights
survived — expected, because `pack` is zerobias + f16 + dead-column pruning
*before* the spline refit, and only the spline half is rank-invariant. The
`bake_dial_refit` owner changed between 2026-09-01 and 2026-09-05 (the ladder
lane's negative-tail work). **Flagged, not fixed**: it is another lane's owner,
the effect is ≤2e-5 on one axis, and the composite is bit-identical.

**Consequence adopted for this campaign:** the G1 gate is read on the RAW
bakes, which is the trainer-equivalence question it exists to answer. Every
arm is still reported from its PACKED verdict, as every prior fast-class cell
was, so arm-to-arm comparisons stay internally consistent.

## 2. A NAMING RESULT THAT FELL OUT OF THE FIRST FIT

The trainer refuses to stamp `zentrain.feature_set_id` on any cell of this
campaign, and it is right to:

```
WARNING: training groups span 2 DIFFERENT feature sets
(basic+peaks+masked+iw+v2+append+append2@w944/era2r4#b782e349 ;
 basic+v2+append+append2@w944/era2r4#7ed470b4)
— refusing to stamp one of them as the bake's zentrain.feature_set_id.
```

That is the fastclass wave's **free-40 train/serve skew** (its AMENDMENT A3.1),
surfacing in the naming layer instead of in a footnote: the base recipe's
`tsafesyn` leg is the only group taken from `foldapp2_views/`, where
`f156..371` are structural zeros, while every other leg is the pools-LIVE root.
So the recipe genuinely trains the 72 peaks on one distribution and serves them
on another, for 1 of its 9 legs.

**The id machinery caught a real defect that prose had already priced as
"bounded and sub-noise" and then moved on.** Not fixed here — swapping the leg
would change the teacher and confound every arm against the incumbent — but it
is now a machine-checkable fact attached to every bake this campaign produces,
which is what fundamental 3 was for.

## 3. THE SHIP BLOCKER IS A7r, AND THE id100 CHAIN CLOSES THE CONTRACT

*(Measured while Phase A was 4 of 30 cells in; only the CONTROL cell had been
read. Both results are on bakes that already existed, plus one re-pack.)*

### 3.1 The id100 chain works on this class, unchanged and with rank untouched

The exact command (recorded here because the commit that first reported it lost
two literals to shell substitution — use a heredoc for messages with backticks):

```sh
bake_dial_refit pack \
  --in  bakes/F2_S265_H128_p_s4004.bin \
  --out bakes/F2_S265_H128_p_s4004_id100.bin \
  --neg-tail \
  --anchor anchors/anchor944_pools_id100.parquet --target-col target_score \
  --verify <root>/ext_cid22val.parquet --verify-col human_score
```

`anchor944_pools_id100.parquet` = `anchor944_pools_dial.parquet` (2,020 rows)
**concatenated** with 21 identity rows at `target_score = 100`, built by
`benchmarks/fastclass2_campaign_2026-09-05/build_id100_anchor.py`. The
concatenation (rather than a second `--anchor-parquet`) is forced: `pack` takes
exactly ONE anchor; only `fit-lasso` accepts a repeated flag. `n_id = 21` is
`d_id100`'s registered value, reused — 1.03 % of anchor mass there, 1.03 %
here.

| | before | after |
|---|--:|--:|
| C5 identity rows outside the band | **39 of 39** | **0** |
| CONTRACT | 5/6 (FAIL) | **6/6 (PASS)** |
| C1 monotonicity | 0.9893662271373883 | 0.9893662271373883 |
| C3 negative-tail frac<0 | 0.8585 | 0.8535 |
| C4 deepest probe dial | −84.4508 | −84.7335 |
| C6 cells above identity | 0 | 0 |
| CID22 (pack verify) | 0.8863 | 0.8863 |

Prune identity gate PASS, all 2,041 anchor scores bit-identical (class 1 only);
944 → 265 layer-0 inputs, caller width unchanged.

### 3.2 A7r: no 944-width model of ANY class passes, and the dial cannot fix it

944 ladder instrument, `--floor-rule resolvable`. A7r = how many of the 5
codecs have a floor-representability fraction below the mentor's own.

| bake | class | **A7r** | contract | C1 mono | C3 | C4 |
|---|---|--:|---|--:|--:|--:|
| **shipped Profile D** | 372 ADD156 additive | **0 — PASS** | PASS | 0.9931 | 0.9145 | −213.15 |
| `Fctl_id100negrich` | 156 slice, 944 additive | 2 | PASS | 0.9879 | 0.7725 | −115.82 |
| `Fpeaks_id100negrich` | 228 slice, 944 additive | 4 | PASS | 0.9628 | 0.7790 | −118.78 |
| `Ffree_id100negrich` | 265 slice, 944 additive | 4 | PASS | 0.9615 | 0.7855 | −138.26 |
| `W11J_s4013` | 944-full MLP leader | 4 | PASS | 0.9902 | 0.0010 | −7.14 |
| `FC_D3_s4004` | the fast-class incumbent | **5** | FAIL (C5) | 0.9398 | 0.8405 | −132.95 |
| `F2_S265_H128_p_s4004` (control) | 944 MLP | **5** | FAIL (C5) | 0.9402 | 0.8585 | −84.45 |
| ↳ same, id100-packed | 944 MLP | **5** | **PASS** | 0.9401 | 0.8535 | −84.73 |

**Only the shipped 372 additive passes, and the id100 chain does not move A7r
by a single codec** — which is the point: A7r is a ladder-*ordering* property of
the weights, and a monotone output spline cannot reorder anything. The d_peaks
lane reached the same conclusion independently at 372 width (*"the raw
pre-spline model is already inverted at the same step — lever is in the fit,
not the spline"*).

**Consequence for this campaign, stated before the arms land:** rank
competitiveness and shippability are now two separate questions with two
different blockers. The plan's ship rule is unrelaxed; A7r becomes a reported
axis on every arm (gate G6) so the answer to "does any set or shape move it?"
is data rather than a single end-of-campaign verdict.
