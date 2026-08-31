# Era-2 items E + F: the drop / redefinition decision table

**Status: decision surface for the user. Nothing here is shipped.**
Assembled 2026-08-31 from four lanes' measurements. Every row states what it
buys, what it costs in rank, and what it takes to ship. Where a number was
measured by another lane it is attributed; where it is not measured at all,
the cell says so rather than being estimated.

The registered bar for a redefinition is era-2 §21.1: **PASS iff no corpus
loses more than 0.005 SROCC and the composite does not fall.** It was
registered before any candidate existed.

---

## The table

| # | Candidate | What it buys (measured) | Per-model rank cost | To ship | Lane |
|---|---|---|---|---|---|
| **E1** | **Drop the v1 masked/IW/soft-peak pool pass** (`V1PoolsMode::Full` → `Off`) | **−41.2 ms = 13.6 %** of the tiled 5 MP walk (2304², 1T: `fold` 78.79 → 37.59) | **944 MLPs: exactly 0.** W-LIN: −0.005 CID22 / −0.048 KonJND. **B: −0.399 CID22 / −0.525 KonJND** | **Per-model, via item D** — it is a compute-set choice, not a global edit. Free for a 944-MLP-only deployment; unshippable for B | cost: era-2 (§24.1); rank: frontier lane, relayed |
| **F1** | **`BLUR_RADIUS` 5 → 4** | Time-**neutral** (+0.68 % 1T / −0.17 % 16T, inside the 4.67 % cross-build layout floor). **Peak RSS −1.35 %** | **PASSES the bar on the shipped 944 flagship**: worst corpus −0.0007, composite **+0.0038**. Fails on exactly one corpus each for three other models, whose composites also rise | **Era** (byte change). No retrain needed for the flagship | blur/radius lane |
| **F2** | **`BLUR_RADIUS` 4 → 3 or 2** | **−4.4 % to −7.1 %** at 16T; radius 2 × `STRIP_ROWS` 32 = **−7.6 %** walk and **−37.6 %** peak RSS at 2304²/1T | **FAILS the bar.** Direction is consistent: gains on cid22/aic3/aic4 and hugely on KonJND (+0.089 at R=2), loses on TID/KADID | **Retrain** (a radius-4 retrain is registered, not launched, and could move this) | blur/radius lane |
| **F3** | **Column-tile the phase-A H blur (packed)** | **1.151× @2304², 1.733× @4608²** at 1T; **1.234× @4608²/8T**, 1.109× @16T; nothing below the tile width (same code path) | **NOT MEASURED.** Same quantities, different summation grouping — needs the rank-preservation gate | **Era** (byte change; running sum restarts per tile) | era-2 (§23) |
| **F4** | **Redirect phase A's `mu2`/`ssq`/`s12` V sweeps into the fold** (exit 1, inner-row copy) | **−3.6 % @2304², −5.1 % @4608²** at 1T, reduced by the `mu2` carve-out on the Y channel | **NOT bit-identical** — 677/956, the V-restart class (the fold V-blurs a band buffer, phase A the wide window). Built and **reverted**: needs stores in all four fold paths + `mu2` kept where BANDVIS runs + **rayon band chunking** so serial and parallel agree | **Era**, plus the parallel-arm work in §29.5 | era-2 (§29) |
| **F4b** | Same, exit 2 (inner-only store offset) | +3.0 ms @2304², +15.7 ms @4608² *on top of* F4 | as F4 | Output row-offset through six `fused_vblur_ssim_inner` tier bodies | era-2 (§24.5) |
| **F6** | **Era-2 accumulation** (fixed 8 virtual lanes) — `ZENSIM_ERA2_DENSE=1` | **Perf-neutral**: 1.026× @2304², 1.015× @4608², inside the noise floor. Its value is the fixed grouping — cross-vendor divergence 66/105 → **0/105** | 242 of 956 slots move, all in `f372+`, max 5.07e-6 relative. Rank-checkable as of this commit; **not yet rank-checked** | **Era**, but **zero test re-pins** — the five goldens all pin v1's 372, which this does not touch | era-2 (§28) |
| **F5** | **Basic-only feature class** (no v2-348 / append) | **2.3–3.6× faster overall** | Not a drop — it is a different model class. B and W-LIN already live there; the 944 MLPs do not | **Model choice**, no code change | blur/radius lane |
| **—** | *Branch/tail-shape work* | **0.14–0.50 % of cycles total**; worst tail class +0.06 pp | — | **Closed — nothing to ship** | blur/radius lane |
| **—** | *Band-local phase A / rolling row window* | **+13.1 % / +3.0 %** (i.e. slower) at R=5 | — | **Closed at R=5.** Sign flips near R=2 — revisit only if F2 ships | era-2 (§22), v2-block (L3) |
| **—** | *Packed column slab* | ceiling **5.4 %**, already reduced by a measured kernel penalty at narrow width | — | **Not built** — premise falsified (the fold is not width-diseased) | era-2 (§24) |

---

## What is shippable TODAY, and what is not

**The shippable-today subset is TILING ALONE (F3).** It passes 5 of 6 models at
the production tile width 1024, and 8 of 9 corpora are byte-identical *by
construction* — every H entry guards `width > tile` and only AIC-3 has refs
wider than 1024. The single FAIL is `BHdr`, missing the bar's zero-tolerance
composite clause by **3.2e-6** (13 % of the `+0.000024` non-event the bar
itself cites) with its worst-corpus loss 125× inside the 0.005 clause. The rank
lane reported that as FAIL rather than renegotiating the bar, which is right —
**the disposition is the user's**, framed as "tiling ships, with one model
3.2e-6 outside a zero-tolerance clause".

**F1 (radius 4) is NOT shippable for the roster today.** It is 2 PASS / 4 FAIL
(C944 and ADD156 pass; B, both W-LINs and BHdr fail) — but every FAIL is an
**upper bound**, because all six bakes were trained at era-1. **Radius 4
becomes shippable for the whole roster only behind the registered retrain**,
which is unlaunched.

**F6 (the accumulation) is rank-checkable as of §28 and not yet rank-checked.**

And the components are **separable, with radius dominating**: every model's
worst-corpus delta is identical to five decimals with the tile on or off, and
the tile moves a composite by ~1e-5 against radius's ~4e-3. **The break's
verdict is really the radius decision.**

---

## What composes with what

The two live speed levers were measured together by the blur/radius lane:
**335.2 → 272.8 (tile) → 255.8 (tile + radius 4) = 1.311× at 2304², and
1.968× at 4608²** — 98 % of the product of the individual factors, so they are
very nearly independent. F4 is measured against an untiled baseline and acts
on `planesA`, which tiling does not touch, so it should add rather than
overlap; **that composition is not yet measured** and should be before any
combined claim is made.

Radius drops **out** of the tile-width grid: tiling is radius-insensitive
(1.229/1.189/1.203× at 2304² for R=5/4/3), one fewer dimension to sweep.

## How to read the rank-cost column

E1 is the row that matters most and it is the one that is **model-conditional
rather than global**. The pool slots cost 13.6 % of every compare. For the
944 MLPs they are worth *exactly nothing* — dropping them changes those models'
outputs not at all. For **B** they are worth 0.399 CID22, which is not a
trade, it is the model. So the correct shipping form is not "drop them" or
"keep them" but **"let the request say"** — which is precisely item D, the
compute-set descriptor. Until D exists, E1 is unshippable in either direction
without picking a winner among models.

F1 is the opposite shape: it is global, it is nearly free in time, and its
value is **peak RSS and the halo** rather than wall clock. It is worth taking
in the break because it is the one redefinition that has *passed* the
registered bar on the shipped flagship — and because it makes every row-shape
question cheaper if F2 is ever revisited.

## What is still missing before any of this can flip

Unchanged from the era-2 doc, and none of it is optional:

1. **Rank preservation across the roster** for every byte-changing row
   (F1 has it; **F3 does not**; F4 may not need it if the edge gate shows
   bit-identity).
2. ~~**Blast radius + retrain wave registration.**~~ **DONE** —
   [`era2_blast_radius_2026-08-31.md`](era2_blast_radius_2026-08-31.md).
   Headline: tiling and the accumulation have **essentially no data-side blast
   radius** (0 re-extraction, 0 retrain); **radius 4 has all of it** (the
   5.74 M-row bigcodec table is ~97 % of the re-extraction, plus 378 board
   cells to rescore under a new era stamp, plus a 6-arm retrain wave).
3. ~~**Gate re-pin enumeration, old → new.**~~ **DONE** for the tiling flip —
   era-2 doc §27: forcing the tile on for the whole suite fails **exactly five
   tests, all absolute-value goldens** (the four `v1_golden_bytes` fixtures and
   `hardcoded_reference_scores`), and **zero** internal-consistency gates. That
   enumeration also found a defect: tiling selected call sites split the v1
   reference path from the fold, so the tile now lives on all four H **entries**
   rather than at call sites — either every H entry tiles or none does.
