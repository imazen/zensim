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
