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

### 3.3 The A7r design, filled in across WIDTH × CLASS (all from bakes that already existed)

The `d_peaks_jxl_floor` lane attributed A7r failure to the **peaks block**:
shipped D (`f0..155` only) is clean on all 33 jxl ladders while *every*
peaks-slice fit it tried inverts on the same 4 images. Crucially, **every model
in that arc is a sparse LINEAR lasso fit — the lane ran no MLP at all**, so
whether an MLP over the same slice inverts was open. Filling the design:

| bake | width | class | slice | **A7r** | C1 mono |
|---|--:|---|---|--:|--:|
| shipped Profile D | 372 | additive | 156 | **0** | 0.9931 |
| shipped D, era-1 dial | 372 | additive | 156 | **0** | 0.9942 |
| `Fctl_id100negrich` | 944 | additive | 156 | 2 | 0.9879 |
| `v47_strict_qat_native` | 372 | **MLP** | 372-full | 4 | 0.9803 |
| `Fpeaks` / `Fpeaks_id100negrich` | 944 | additive | 228 | 4 | 0.9604 / 0.9628 |
| `Ffree_id100negrich` | 944 | additive | 265 | 4 | 0.9615 |
| `W10L9PH_s4006` / `W11J_s4013` | 944 | MLP | 944-full | 4 | 0.9849 / 0.9902 |
| shipped Profile B | 372 | additive | 372 linear | 5 | 0.9776 |
| `A3b_s4004` | 944 | MLP | 265 | 5 | 0.9419 |
| `FC_D3_s4004` (incumbent) | 944 | MLP | 265 | **5** | 0.9398 |

**Neither "372" nor "additive" is the explanation.** Shipped Profile *B* is a
372-width additive fit and reads **5**; `v47` is a 372-width MLP and reads 4.
The single scorer that passes is shipped Profile D, whose distinguishing
property is the **156-basic-only slice** — and the one other 156-only model in
the table (`Fctl`, the same slice at 944 width, a different fit) is the next
best at **2**. Everything that adds coordinates beyond `f0..155` — peaks,
moments, the full v2 block, class-C — sits at 4 or 5 regardless of width or
class.

That is consistent with the jxl-floor lane's attribution and **extends it to
the MLP class it never tested**: this campaign's `S156` arm (an MLP on
`f0..155` at 944 width, k = 3) is the decisive cell, and it is already in Phase
A.

### 3.4 A THIRD blocker, and it is wiring rather than physics

`Zensim::compute()` emits a **372**-layout vector, so a 944-declared-width bake
is refused with `ModelForwardFailed { reason: "bake declares more input
features than the caller supplied" }` — the D+free lane hit this on
`Fpeaks_id100negrich` and it applies to **every candidate this campaign
produces**.

**It is not an impossibility, and the distinction matters.** The 944-layout
walk + forward already runs today in this repo's own speed instrument:
`zensim-bench/benches/ssim2_speed_bar.rs`'s `free156_peaks_raw` arm loads an
`A3b`/`A4b`-class 944-width bake through `ZEN_HY_FREE` and forwards it over
`compute_folded720_features_streaming(.., v1_basic_free, ..)`. What is missing
is a `ZensimProfile` wired to that walk — the W7 clause the fastclass wave's
§6e already named.

**Two ship paths follow, and only one needs new code:**

1. **Wire a 944-emitting profile** (the W7 task). Serves any of S156/S228/S261/
   S265/S289 as trained.
2. **Refit the winner at 372 width**, which is possible *only* for S156 and
   S228 — `f0..227 ⊂ f0..371`, whereas S261/S265 need `f733+` and S289 needs
   `f377+`. This is exactly the move the d_peaks lane made
   (`Dpeaks372_id100negrich_dial_fsid.bin`, 20 basic + 6 peaks coefficients at
   f162-164/f211-212/f224). Its prerequisite is a 372-wide, era-consistent
   version of this recipe's training legs: the 372 roots carry the eval corpora
   but the big legs exist only at older eras (`tbig_372_200k.parquet`,
   `2026-05-1x` safesyn), so assembling one is a corpus job, not a flag.

**Recorded, not attempted.** Both are named here with their prerequisites so
the ship decision is a choice rather than a discovery.

### 3.5 Every A7r failure in this population is an ORDERING inversion — zero are clamps

Per-codec detail, read from the same gradings (`repr` = fraction of
`(image, codec)` ladders represented; `mentor` = ssim2's own on the same cells;
`unord` = ladders failing on ordering; `clamp` = ladders failing by sitting on
the instrument minimum):

| bake | avif-rav1e | avif-svt | jpeg | jxl | webp | unord total | clamp total |
|---|---|---|---|---|---|--:|--:|
| **shipped D** | 0.6667 ✓ | 1.0000 ✓ | 0.6667 ✓ | 1.0000 ✓ | 1.0000 ✓ | 26 | **0** |
| `Fctl_id100negrich` (156, additive) | 0.4103 ✗ | 1.0000 ✓ | 0.6667 ✓ | 0.9615 ✓ | 0.9744 ✗ | 25 | **0** |
| `Fpeaks_id100negrich` (228, additive) | 0.1282 ✗ | 1.0000 ✓ | 0.5641 ✗ | 0.6923 ✗ | 0.9231 ✗ | 62 | **0** |
| `W11J_s4013` (944 MLP) | 0.5128 ✗ | 0.9487 ✗ | 0.5385 ✗ | 0.9615 ✓ | 0.9744 ✗ | 41 | **0** |
| `v47_strict_qat_native` (372 MLP) | 0.3590 ✗ | 0.8462 ✗ | 0.5128 ✗ | 0.8462 ✗ | 1.0000 ✓ | 54 | **0** |
| **`CTL_ID100`** (the fast-class control) | 0.1538 ✗ | 0.5897 ✗ | 0.5385 ✗ | **0.3077** ✗ | 0.7949 ✗ | **93** | **0** |
| *mentor `peer_ssim2`* | *0.6410* | *1.0000* | *0.6667* | *0.9615* | *1.0000* | — | — |

Three things follow, none of them previously on the record:

1. **`n_fail_clamp` is 0 for every bake and every codec.** A7r in this
   population is exclusively about *ordering the bottom of a ladder*, never
   about a dial pinned to its floor. So the levers are the ones that act on
   ordering — the fit, `--monotonicity-reg`, `--monotone-cbc` — and not
   anything that changes the dial's range.
2. **The bar is relative and the mentor is itself weak in two places**: ssim2
   represents only 0.6410 of `avif-rav1e` and 0.6667 of `jpeg` ladders. Shipped
   D passes `jpeg` by matching the mentor *exactly* (0.6667 vs 0.6667). The
   task is "no worse than ssim2", not "perfect".
3. **The fast class's worst codec is `jxl` (0.3077 against a 0.9615 mentor, 18
   of 26 ladders unordered)** — a much deeper failure than the additive
   peaks-slice fits the d_peaks lane studied (0.6923, 8 of 26). So the MLP does
   not merely inherit the linear class's jxl inversion; it is markedly worse on
   the same codec, which is a new fact about the model class rather than about
   the slice.

### 3.6 What the four LUMA_MEAN_REF slots are worth, measured BEFORE the S261 arm lands

`inspect_l0_input_norms --top 944` on the two control bakes that exist (L2 norm
of each layer-0 input column; the two printed tables deduplicated by index):

| block | slots | s4004 mass | s4005 mass |
|---|--:|--:|--:|
| basic `f0..155` | 156 | 56.32 % | 57.20 % |
| peaks `f156..227` | 72 | 33.86 % | 34.25 % |
| raw moments `f733..922` | 33 | 9.18 % | 8.10 % |
| **`LUMA_MEAN_REF` f926/931/936/941** | **4** | **0.64 %** | **0.45 %** |

Their per-slot L2 ranks are **238–260 of 944** (norms 0.90–0.93 and 0.61–0.62
against a max of 9.04 / 8.83). **So the identity fix of §0(a) is predicted to
cost about half a percent of layer-0 weight mass** — a prediction the `S261`
arm tests directly against `S265` at k = 3, rather than an assumption the slice
file was written on.

Second reading, worth recording on its own: in this MLP the **peaks block
carries 34 % of the mass** against the 19.6 % the D+free lane measured in the
sparse *linear* fit on the same slice, and the raw moments carry 8–9 % against
2.3 %. The fast class's MLP leans considerably harder on the non-basic half
than the additive class does — which is the same direction as its much worse
A7r (§3.5) and is a hypothesis the `S156`/`S228` arms can falsify.
