# Attribution append2 slice (f924-943) — E1 determination, fix, and M3a impact

Registration: `benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX E
(committed at `27168fe2` BEFORE any M3a was re-measured). This document reports
the results against that registration.

Defect found 2026-08-04 by the coherence study (`0c11f52c`):
`zensim/src/attribution.rs` sliced the model gradient as
`s[720..s.len().min(924)]`, so on a **944-input** model the whole append2 /
BANDVIS block was dropped from the attribution density. Previously unnoticed
and undocumented — it silently applied to every 944-era bake.

---

## 1. What f924-943 are, and which are spatializable (derived from source)

The registration's §E.1 question was deliberately *not* "is the slice bound
wrong" but "what would an honest per-pixel integrand for each slot be". Layout:
`f924 + scale*APPEND2_PER_SCALE + local`, `APPEND2_PER_SCALE = 5`, 4 scales,
**Y-only** (no channel axis) = 20 slots. Definitions in
`zensim/src/feature_v2.rs` `idx_append2`; accumulation in the gradient and
append kernels; finalize in the append2 block of `finish_*`.

| local | slot | production pooling | spatializable | class |
|---|---|---|---|---|
| 0 | `BANDVIS_GAIN` | `clamp01(Σ gain_i / N)`, `gain_i = bounded_excess_pair(b_dst, b_src, C_BV).0`, `b_x = band(curv_x)·flat` | **YES** | **E** |
| 1 | `BANDVIS_LOSS` | same, `.1` | **YES** | **E** |
| 2 | `LUMA_MEAN_REF` | `sat(mean(ref Y), C_LUM_T)` — reference-only | **NO** | **N by definition** |
| 3 | `HL_BIN1` | `Σw·mse_i/Σw`, `w = sat(max(y_ref − HL1_Y_ANCHOR, 0), C_HL)` | form yes, route no | **N (structural zero)** |
| 4 | `HL_BIN2` | same, `HL2_Y_ANCHOR` | form yes, route no | **N (structural zero)** |

**Evidence per row.**

- **BANDVIS_GAIN/LOSS are class E.** `gain_i`/`loss_i` are per-pixel indicator
  values and the feature is their **plain mean over the plane** — bit-for-bit
  the pooling form the v2 `HF_GAIN` / `HF_LOSS` / `HF_MAG_LOSS` slots already
  carry as class E ("mean of bounded_excess", integrand `−v_i/N`). Nothing new
  had to be invented; the C2a construction transfers verbatim. **Measured
  confirmation:** the full-plane density sum reproduces the production 944
  feature to 8–9 significant digits at every scale (§4).
- **LUMA_MEAN_REF is N by definition.** It is `sat(mean(ref Y))` off the
  append kernel's reference-side `Σs` accumulator. `∂f/∂(distorted) ≡ 0`, so a
  zero density is the exactly-correct answer — the same footing as v2
  `PJND_FRAGILITY` and append `GRAD_SRC_MEAN`, which likewise carry no term.
  Its own doc comment already said "correct-0 in any steering fold".
- **HL_BIN1/2 are N on this route.** Their *form* is class E (identical to the
  append luminance bins `LUM_DARK/MID/BRIGHT_ERR`), but they are accumulated
  only under the HDR const-generic `HL`, and the attribution path is
  structurally SDR: `attr_pass_a_kernels` passes `hl = false`
  (`// hl (append2) — not part of the 924 regime`), and both sides come through
  the SDR `prepare_v2_reference_impl`. On the SDR route `Σw ≡ 0` ⇒
  `WeightedSum::finish() ≡ 0` ⇒ the feature is identically `0.0` ⇒ `Δf ≡ 0`
  for any probed gradient. Same footing as the X/B transducer slots and the
  `APPEND_SKIP_B_SCALE0` cell.

**Verdict: the slice was BOTH a genuine defect AND correct — per slot.** 8 of
the 20 slots (BANDVIS gain+loss × 4 scales) were real, dropped coverage. The
other 12 were correctly zero, but *silently* so — by a slice bound rather than
by an integrand. Both halves are now fixed: the first by construction, the
second by explicit naming.

## 2. What changed

- `Zensim::compute_attribution_density_full` slices a fourth block
  `s[924..min(len, 944)]` and threads it into
  `feature_v2::compute_v2_append_attribution` as `s_append2`.
- Named `BLOCK_END_{BASIC,V1_POOLS,V2,APPEND,APPEND2}` constants replace the
  magic bounds, cut by one shared half-open `block(start, end)` helper.
- `V2AppCoeffs` gains `c_bv_gain` / `c_bv_loss` (`−s_k/N`, the existing
  mean-pool convention). `derive_v2app_coeffs` sets them on the Y channel
  only (`APPEND2_CHANNEL`), with the per-slot classification written out in
  the code as a comment so the three zero slots are a stated decision.
- The pass-B **gradient family** loop computes the BANDVIS terms in f64 from
  the same cached planes. The second differences (`d2x = x_l + x_r − 2·x`)
  **reuse the four neighbour loads the loop already performs** for the
  gradients, so the terms are near-free, and the neighbour convention (x:
  clamp, y: `reflect_101` via the production halo rows) matches the kernel
  exactly. `V2NewFeatureToggles::default()` has `append2_dst_activity = false`,
  which is also the adjudicated production setting, so the OFF-toggle form is
  the correct — and only — one to replicate.

Nothing outside the 944 width can change: 372/720/924 gradients never reach
924, which the coverage test now asserts.

## 3. Tests added (three gates)

1. **`attribution::tests::attribution_covers_expected_slots_per_width`** — the
   anti-recurrence guard. For widths 372 / 720 / 924 / 944 it probes one slot
   at a time and asserts the density is non-zero exactly where the table above
   says it must be and identically zero where it says N — including that
   f944+ (CSFW) is deliberately, explicitly not covered yet. Adding a block
   now requires adding a `BLOCK_END_*` constant **and** a row here.
2. **`feature_v2::tests::append2_attr_sum_identities_and_zero_slots`** — the
   class-E claim: full-plane density sum vs the PRODUCTION 944 feature
   (`compute_folded720_append2_features`) at every scale where the feature is
   non-trivial, at the same 1e-5 class as the existing C2a sum-identity gate;
   plus `|mass| == 0.0` exactly for `LUMA_MEAN_REF` / `HL_BIN1` / `HL_BIN2` at
   every scale. It fails loudly (rather than passing vacuously) if the fixture
   stops exercising BANDVIS.
3. **`feature_v2::tests::append2_attr_bandvis_fd_direction`** — the FD
   direction gate, the C2a precedent that caught the edge-width sign bug.

### The FD-region finding (worth recording)

The FD test first used a compact centred block (96×96 on 160×160) and
**failed** at scale 2 on `BANDVIS_LOSS`: predicted −0.00732 vs true
**+0.00039** — a sign flip. That is not a formula error; it is the documented
finite-removal floor, and the geometry explains it. The integrand is exact for
pixels whose 3-tap curvature stencil stays inside the refined set, so the
error scales with the **seam-adjacent fraction**, which grows as `2^scale`:

| refined region | seam-adjacent fraction at scale 2 |
|---|---:|
| compact 96×96 block | ≈ 17 % |
| left **half-plane** (80×160) | ≈ 5 % |

A half-plane is the minimum-seam region for its area (one interior seam; the
other three sides are image edges, which reflect-pad rather than abut
unrefined content). Switching to it, **all 8 slots pass with the correct
sign**, and the exactness (plane-sum) assertion holds at every scale:

| scale | slot | a0 (feature) | true Δf | predicted | pred/true |
|---:|---|---:|---:|---:|---:|
| 0 | GAIN | 0.021918 | −0.010840 | −0.011076 | 1.02 |
| 0 | LOSS | 0.045162 | −0.023328 | −0.023791 | 1.02 |
| 1 | GAIN | 0.011018 | −0.004300 | −0.004882 | 1.14 |
| 1 | LOSS | 0.027793 | −0.013626 | −0.015256 | 1.12 |
| 2 | GAIN | 0.005005 | −0.001363 | −0.002039 | 1.50 |
| 2 | LOSS | 0.018448 | −0.008001 | −0.010116 | 1.26 |
| 3 | GAIN | 0.003436 | −0.000464 | −0.001753 | 3.78 |
| 3 | LOSS | 0.016850 | −0.003756 | −0.008924 | 2.38 |

The monotone drift up the pyramid is the approximation the module already
documents; a formula error shows up as a sign flip or an order-of-magnitude
miss. The **exact** claim — Σdensity = −feature — holds at 8–9 significant
digits at **every** scale, independent of any refinement geometry, and is
asserted in both tests.

## 4. M3a impact — MATERIAL

Method (registered §E.3): the same 27 fixtures, same bake bytes, same machine,
serial, scored twice — once with the binary built at the parent commit
(`408dd3c0`, "OLD") and once with the fix ("NEW"). Instrument:
`scripts/m3a_sweep.sh --grid full` (the newly-extracted owner of the grid loop
`run_full_eval.sh` used to inline). Per-cell TSVs: `~/tmp/attrfix/{old,new}/`.

| bake | M3a OLD | M3a NEW | ΔM3a | tier OLD | tier NEW |
|---|---:|---:|---:|---|---|
| `H_co3abpg_s2507` | 0.866393 | 0.889959 | **+0.023566** | GOLD | GOLD |
| `C_em944_s31` | 0.792589 | 0.874867 | **+0.082278** | silver | **GOLD** |
| `C_co3a_s1301` | 0.759778 | 0.786074 | **+0.026296** | — | **silver** |
| `C_co3a_s1307` | 0.762467 | 0.866978 | **+0.104511** | — | **GOLD** |
| `C_ensk2_s1303` | 0.826226 | 0.822296 | −0.003930 | silver | silver |

`M3_MEAN` is byte-identical in all five (0.219096 / 0.206222 / 0.079019 /
0.251648 / 0.002515) — correct: M3 is the legacy signal fold and the fix does
not touch it. Only M3a moves, which is the expected blast radius.

**All three registered materiality triggers fire.** `max |ΔM3a| = 0.1045` ≫
the 0.005 threshold; two bakes cross the **0.85** gold bar; two cross the
**0.78** silver tier. **Three of five change tier.**

**So yes — the campaign's published 944 M3a values shift, and they shift
upward.** Every 944-width M3a measured before this commit understates the
model's true attribution coherence, because the 8 spatializable slots — just
**0.85 % of the 944 layout** — were being discarded before the map was built.
The direction is not a surprise; the magnitude (up to +0.10, more than twice
the 944-class M3a sd of 0.0471) is.

### The block's gradient mass does not explain the size of the shift

`diffmap_block_coherence` prints each bake's raw-`|s_k|` mass on the append2
block. Across all 32 bakes it is **≤ 0.4 %** (the instrument prints one
decimal), yet ΔM3a reaches **+0.10**. So a block carrying well under one
percent of the model's gradient mass was, by itself, worth up to a tenth of
the coherence statistic.

That is not a surprise in this codebase — it is the same shape C2a measured
when the append block alone cured the 128 px inversion ("the 0.5 %-mass
append block was the whole coarse signal"). It is also a third, independent
line of support for §D.8's falsification: **where a bake's contribution mass
sits does not determine its coherence.**

**Do NOT read a mass → ΔM3a relationship out of this.** The printed mass has
one decimal, and several bakes that print `0.0 %` moved by +0.05 to +0.10, so
the precision cannot support one. The single clean observation is that
`sota944_winner_A_bvls_X_AM5` — the one bake whose ΔM3a is **exactly
0.0000** — also prints `0.0 %`; consistent with the mechanism, but one point.

### The defect was materially corrupting SELECTION — a worked example

M3a became a first-class selection input the same day (`92a23417`,
`freeze_check --select`, campaign appendix E.4). Running the registered rule
over the `C_co3a` k = 6 seed family — the campaign's own multi-seed family —
before and after the fix **changes the winner**:

| rank | bake | floors | M3a pre-fix | M3a corrected | sel_comp pre-fix | sel_comp corrected |
|---:|---|---:|---:|---:|---:|---:|
| — | `C_co3a_s1301` | 7/8 | 0.7598 | 0.7861 | **0.9195 (1st)** | 0.9234 (2nd) |
| — | `C_co3a_s1307` | 7/8 | 0.7625 | 0.8670 | 0.9149 (2nd) | **0.9306 (1st)** |
| — | `C_co3a_s1409` | 7/8 | 0.7181 | 0.7367 | 0.9076 (3rd) | 0.9104 (3rd) |

`s1301` and `s1307` pass the same 7 of 8 floors, so the decision falls to the
tie-break — and `s1307` gained **+0.1045** M3a to `s1301`'s **+0.0263**. The
pre-fix ranking picked `s1301`; the corrected ranking picks `s1307`.

Two things worth stating plainly:

1. **The bug was not cosmetic.** Left in place, it would have silently biased
   every k-seed selection from the day the rule shipped — toward whichever
   seed happened to lean least on the block the map was discarding.
2. **The primary still governs.** `C_co3a_s1319` has the highest corrected
   M3a in the family (0.8786) and the second-highest `selection_composite`,
   and still ranks **4th**, because it passes only 6 of 8 floors. Coherence
   breaks ties; it does not override a failed floor. That is the registered
   rule behaving as designed, on real data.

### What this does NOT invalidate

- **Nothing at 372 / 720 / 924 width.** Those gradients never reach f924. The
  §D.8 coherence-mechanism analysis mixes widths, and its four named
  counterexamples are 372/720/944 — the 944 rows move, so §D.8's *numbers*
  need re-derivation before being cited again, but its structural
  counterexamples at 372/720 stand unchanged.
- **No rank, dial, corruption or breadth number** — M3a is a reported tier in
  the balanced profile, not one of its floors (`M3A_GOLD`/`M3A_SILVER` are
  explicitly "reported tier, NOT a floor"). No `freeze_check --profile
  balanced-2026-08-04` PASS/FAIL verdict changes as a result of this fix.
  It DOES change the §5 freeze-bar M3a row (which IS a bar), and — because
  M3a entered the selection rule the same day (appendix E.4) — it changes
  `freeze_check --select` tie-breaks. Both consume the corrected values.

### Re-measurement of the full population

Per the registration's MATERIAL branch, **every 944-width board cell carrying
an M3a was re-measured with the fixed binary and its fulleval updated in
place**, through the committed owner rather than a hand-rolled injector:
`run_full_eval.sh` gained `ZENSIM_M3_ONLY=1` (the inverse of the existing
`ZENSIM_M3_REUSE`) which skips `bake_verdict` entirely and refreshes only the
M3/M3a keys. Verified on a scratch copy before the pass: exactly four keys
change (`m3_coherence`, `m3_n`, `m3a_coherence`, `m3a_n`) and all 17 other
top-level keys are byte-identical, so no rank/dial/corruption number is
touched. `promote_fulleval.py`'s "carry m3a where measured" rule then
preserves the corrected value through any later re-promotion.
`benchmarks/eval_annotations.json` carries the `kind=invalidated` entry
`m3a-pre-append2-fix`, scoped BY NAME to exactly these 32 cells — not to
every `m3a_coherence` on the board, which would have falsely flagged the
unaffected 372/720/924 values. Results and the full old→new table: §6.

## 5. Cost of the M3a instrument, and the cheap-grid verdict (registered §E.5)

**Cost.** `run-heavy`-supervised, serial, `H_co3abpg_s2507`, 27 cells:
**66.3 s wall** (user 1 m 58 s, sys 37 s) — **below the registered 120 s/bake
trigger**, so the full instrument is kept and there was never a cost case for
the cheap variant.

**Agreement, measured anyway.** The registered 9-cell balanced Latin square is
a strict SUBSET of the full grid, so its value is derivable from the *same*
per-cell measurements — no re-measurement, no run-to-run confound
(`scripts/v_next/m3a_cheap_grid_agreement.py`). Over the full 32-bake
population:

| statistic | measured | registered gate | |
|---|---:|---|---|
| SROCC(cheap, full) | **0.8871** | ≥ 0.90 | **FAIL** |
| max \|cheap − full\| | **0.1021** | ≤ 0.02 | **FAIL** |
| mean \|cheap − full\| | 0.0193 | — | |

Worst cells: `C_co3a_s1409` +0.1021, `C_co3a_s1303` −0.0508, `C_ensk2_s1301`
−0.0328. **Both halves fail, and the magnitude is the point:** 0.1021 is more
than **twice the whole 944-class M3a sd (0.0471)** — a cheap-grid M3a can move
a bake further than the entire signal being selected on. The 27-cell mean is
doing real variance reduction over a wide per-cell spread; cutting to 9 keeps
the design's *balance* but not its *precision*.

**Executed outcome:** the full instrument is the only one shipped.
`m3a_sweep.sh --grid cheap` is a hard ERROR that prints these numbers, so the
rejection is stated where someone would reach for it; the subset definition
and the measurement live on in the analysis script, which derives the subset
from full-grid TSVs and needs no support in the sweep.

## 6. Results of the full re-measurement

### 6.1 Full 944-width population — old vs new (n = 32)

Every board cell at `n_inputs == 944` that carried an `m3a_coherence`. Each bake's `.bin` was on disk and its **sha256 matched the `bake_sha256` recorded in the fulleval** before scoring (32/32), so each before/after pair is the same bytes through two binaries. The NEW column is read back from the **updated board fulleval**, so this table and the board cannot disagree.

| bake | M3a OLD | M3a NEW | ΔM3a | tier OLD → NEW |
|---|---:|---:|---:|---|
| `C_co3a_s1307` | 0.7625 | 0.8670 | +0.1045 | flagged → GOLD  **↑** |
| `C_co3a_s1307_packed` | 0.7626 | 0.8669 | +0.1042 | flagged → GOLD  **↑** |
| `C_co1a_s1307` | 0.7869 | 0.8893 | +0.1024 | silver → GOLD  **↑** |
| `C_ensk2_s1301` | 0.7823 | 0.8679 | +0.0855 | silver → GOLD  **↑** |
| `C_em944_s31_packed` | 0.7924 | 0.8750 | +0.0826 | silver → GOLD  **↑** |
| `sota944_C_em944_s31` | 0.7926 | 0.8749 | +0.0823 | silver → GOLD  **↑** |
| `C_ensk5_s1303` | 0.7934 | 0.8745 | +0.0811 | silver → GOLD  **↑** |
| `C_co4_s1303` | 0.8352 | 0.8988 | +0.0636 | silver → GOLD  **↑** |
| `C_ensk5_s1301` | 0.7849 | 0.8465 | +0.0616 | silver → silver |
| `C_co1a_s1303` | 0.7713 | 0.8265 | +0.0551 | flagged → silver  **↑** |
| `C_co4_s1307` | 0.8035 | 0.8581 | +0.0546 | silver → GOLD  **↑** |
| `C_co1b_s1303` | 0.7932 | 0.8467 | +0.0535 | silver → silver |
| `C_co3a_s1319` | 0.8259 | 0.8786 | +0.0526 | silver → GOLD  **↑** |
| `C_co2a_s1307` | 0.8261 | 0.8785 | +0.0525 | silver → GOLD  **↑** |
| `C_co2b_s1307` | 0.7993 | 0.8494 | +0.0502 | silver → silver |
| `H_co3abpg_s2501` | 0.8280 | 0.8772 | +0.0492 | silver → GOLD  **↑** |
| `C_co4_s1301` | 0.8237 | 0.8719 | +0.0482 | silver → GOLD  **↑** |
| `C_co1c_s1301` | 0.7962 | 0.8443 | +0.0481 | silver → silver |
| `H_co3abpg_s2503` | 0.7735 | 0.8190 | +0.0455 | flagged → silver  **↑** |
| `C_co3b_s1303` | 0.8470 | 0.8911 | +0.0440 | silver → GOLD  **↑** |
| `C_ensk2_s1307` | 0.8053 | 0.8430 | +0.0377 | silver → silver |
| `C_co3a_s1303` | 0.7699 | 0.8075 | +0.0376 | flagged → silver  **↑** |
| `C_co3a_s1301` | 0.7598 | 0.7861 | +0.0263 | flagged → silver  **↑** |
| `C_co3a_s1301_w4repro` | 0.7598 | 0.7861 | +0.0263 | flagged → silver  **↑** |
| `sota944_nt223` | 0.6970 | 0.7210 | +0.0240 | flagged → flagged |
| `H_co3abpg_s2507_packed` | 0.8665 | 0.8904 | +0.0239 | GOLD → GOLD |
| `H_co3abpg_s2507` | 0.8664 | 0.8900 | +0.0236 | GOLD → GOLD |
| `C_co3a_s1409` | 0.7181 | 0.7367 | +0.0185 | flagged → flagged |
| `C_co3a_s1321` | 0.8148 | 0.8325 | +0.0177 | silver → silver |
| `C_ensk5_s1307` | 0.8077 | 0.8223 | +0.0145 | silver → silver |
| `sota944_winner_A_bvls_X_AM5` | 0.6299 | 0.6299 | +0.0000 | flagged → flagged |
| `C_ensk2_s1303` | 0.8262 | 0.8223 | -0.0039 | silver → silver |

**Δ summary:** median **+0.0487**, min -0.0039, max **+0.1045**; 30 of 32 moved up. **19 of 32 cells change M3a tier** — 14 newly cross the **0.85 GOLD** bar and 7 newly cross the **0.78 silver** tier.

**Determinism replicate.** The board values above come from a second, independent pass (`run_full_eval.sh ZENSIM_M3_ONLY=1`) over the same fixtures; the standalone sweep's earlier pass agrees **BIT-IDENTICAL** across all 32 bakes. So the instrument is exactly reproducible run-to-run, and every Δ in the table is the fix, not measurement noise.

Two internal consistency checks fall out of the population and both hold: `C_co3a_s1301` and its independent training-level repro twin `C_co3a_s1301_w4repro` land on the **same** new M3a, and both packed twins reproduce their parents to ~4 dp (`C_co3a_s1307`/`_packed`, `H_co3abpg_s2507`/`_packed`) — matching the campaign's prior finding that packing is M3a-neutral.

## 7. Limitations (complete)

1. **The HL bins are unverified as integrands.** `HL_BIN1`/`HL_BIN2` are class N
   here only because this attribution route is structurally SDR. Their *form*
   is class E, so if an HDR attribution route is ever built they become real
   dropped coverage and must be spatialized then. The coverage test pins today's
   behaviour, not that future one.
2. **First-order, like the rest of C2a.** The BANDVIS integrands inherit every
   approximation the module documents — blur bleed unmodeled, finalize clamps
   treated as inert. The exact claim is the plane-sum identity (verified to 8-9
   significant digits at every scale); the *block* claim degrades with the
   seam-adjacent fraction, measured at pred/true 1.02 → 3.78 from scale 0 to 3.
3. **The FD magnitude band is factor-4.** That is wide. It catches sign errors
   and order-of-magnitude errors, which is what it is for; it would not catch a
   systematic 2× coefficient error. The plane-sum identity is the tight gate.
4. **No pruned bake was exercised end-to-end.** The append2 coverage tests and
   the pruning-width regression test both pass, but no packed-with-pruning 944
   bake existed on disk to run the full harness against — the two fixes are
   verified separately, not composed.
5. **§D.8's coefficients are not re-derived.** The coherence-mechanism
   correlations were computed on pre-fix M3a for the 944 rows. §E.8 restates the
   *conclusions* that rest on them and flags which need re-derivation; the
   coefficients themselves are left as the record of what was measured then.
6. **Wave-6 arm F's paired-lift magnitudes are open.** Its "+0.023..+0.056 in
   6/6 seed-paired draws" compares students to counterparts outside this
   population, which were not re-measured. §E.8.5 corrects only the "max below
   the bar" half.
7. **The cheap-grid rejection is measured on one population.** SROCC 0.8871 /
   max 0.1021 come from the 32 944-width board cells. A different population
   could differ — but since the observed error already exceeds twice the class
   sd, a re-examination would need to beat a wide margin.
8. **`m3a_sweep.sh` has no unit test.** It is shell that shells out; it is
   covered only by having produced every number in this document and by
   `just lint-scripts`. The statistics it reports are computed in Rust.
9. **Self-inflicted, recorded so the next session avoids it:** the first board
   pass lost one bake to a syntax error because a script was edited *while a
   loop was executing it* (bash reads a script incrementally). Do not edit
   `run_full_eval.sh` / `m3a_sweep.sh` during a population run; the cell was
   re-run.
