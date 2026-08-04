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
model's true attribution coherence, because ~4 % of the 944 layout (and, per
the E-M9/C2a finding, a disproportionate share of the *coarse-scale* signal —
which is exactly where the 128 px inversion lived) was being discarded before
the map was built. The direction is not a surprise; the magnitude (up to
+0.10, more than twice the 944-class sd of 0.0471) is.

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
  It DOES change the §5 freeze-bar M3a row, which is a bar.

### Re-measurement of the full population

Per the registration's MATERIAL branch, every 944-width board cell carrying an
M3a is re-measured with the fixed binary and its fulleval updated through the
committed promoter, with an `eval_annotations.json` entry (`kind=invalidated`)
pointing the superseded numbers at this commit. Results and the full old→new
table are in §6 below.

## 5. Cost of the M3a instrument (registered §E.5)

`run-heavy`-supervised, serial, `H_co3abpg_s2507`, 27 cells:
**66.3 s wall** (user 1 m 58 s, sys 37 s). That is **below the registered
120 s/bake trigger**, so per the registration the **full 27-cell instrument is
kept** and the cheap variant is not shipped as the default.

The registered 9-cell balanced Latin square is implemented anyway as
`m3a_sweep.sh --grid cheap` (it was frozen in the registration before any
agreement number existed, so it cannot be tuned to agree), and its measured
agreement against the full grid is reported in §6 so a future session that
does want a 3× cheaper per-seed screen starts from data rather than a guess.

## 6. Results of the full re-measurement

See the tables appended below by the re-measurement run.
