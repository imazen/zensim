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

### 3.7 A7r = 5 survives SEVEN recipe variants — it is not the base recipe's pairing

Grading every fastclass-wave arm (seed 4004, the wave's own packed bakes) on the
same instrument:

| arm | mechanism | A7r | avif-rav1e | avif-svt | jpeg | jxl | webp |
|---|---|--:|--:|--:|--:|--:|--:|
| `C0` | uniform pairing (= A4b) | **5** | 0.2308 | 0.6923 | 0.3077 | 0.3077 | 0.9744 |
| `D1` | kon within-ref | **5** | 0.2051 | 0.5897 | 0.3590 | 0.4615 | 0.8718 |
| `D2` | hf within-ref | **5** | 0.1026 | 0.9231 | 0.5385 | 0.5000 | 0.9487 |
| `D3` = this campaign's base (`CTLPROBE`) | both | **5** | 0.1538 | 0.5897 | 0.5385 | 0.3077 | 0.7949 |
| `D4` | high-q-boost 3.0 | **5** | 0.1538 | 0.6154 | 0.3846 | 0.3462 | 0.8974 |
| `F1` | KADIS w=0.15 | **5** | 0.1538 | 0.6667 | 0.4103 | 0.3846 | 0.9231 |
| `G1` | class-C, 289 coords | **5** | 0.2564 | 0.6667 | 0.5128 | 0.3846 | 0.9487 |
| *mentor `peer_ssim2`* | — | — | *0.6410* | *1.0000* | *0.6667* | *0.9615* | *1.0000* |

**Seven recipe variants, five codecs each, and not one cell clears the mentor.**
So the base-recipe choice this campaign registered (§4 of the plan, the
within-ref pairing) is not what causes the A7r failure — the uniform-pairing
control is equally bad, and on `avif-rav1e` slightly better while on `jpeg`
slightly worse. Neither zone mass, nor a corpus addition, nor the 24 class-C
slots moves it.

**That empties the RECIPE axis on A7r and leaves exactly two untried levers,
both already registered as arms**: the input SET (`S156`/`S228` — the 156-only
slice is the one property the single passing scorer has, §3.3) and the
per-sample-α head's `--monotonicity-reg`, which is the only mechanism in the
trainer that penalises a *pair whose predicted ordering disagrees with the
target's* — which is literally what A7r measures (§3.5: every failure is an
ordering inversion, zero clamps).

## 4. A SILENT NO-OP FOUND WHILE SCOPING PHASE C — `--coarse-decay` on the alpha head

**MEASURED by reading the owner, 2026-09-05, before any alpha-head fit.**
`--coarse-decay` is applied by `zensim_validate::mlp_train::
apply_post_adam_penalties`. That function is called at **seven sites, every one
inside `train_mlp_strategy`'s plain loop**;
`train_mlp_per_sample_alpha_head` (`mlp_train/mod.rs` 6355..10240) contains
**zero** occurrences of it, of `apply_coarse_decay`, or of `coarse_decay`:

```sh
grep -n 'apply_post_adam_penalties' zensim-validate/src/mlp_train/mod.rs   # 7 sites, all in train_mlp_strategy
awk 'NR>=6355 && NR<=10240 && (/apply_post_adam_penalties/ || /coarse_decay/ || /apply_coarse/)' \
    zensim-validate/src/mlp_train/mod.rs                                    # no output
```

**The repo already knew this and had guarded the other rider.**
`group_l1_unsupported_flag`'s own doc comment says it plainly — *"`--group-l1`'s
per-step group-lasso prox (`apply_post_adam_penalties`) is invoked ONLY inside
the plain path's training loop — `train_mlp_per_sample_alpha_head` never calls
it"* — and `--group-l1` has refused `--per-sample-alpha-head` since it landed.
`--coarse-decay` rides the *same function* and was never guarded.

**Why it matters here specifically:** `--coarse-decay 1e-5` is in the fast-class
base recipe, and this repo's CLAUDE.md records it as *"KonJND +0.15, CSIQ +0.07,
~free"*. So a Phase C alpha-head arm run verbatim would have differed from its
control by **two** things — the head *and* a silently-dropped regularizer — while
its embedded `zentrain.repro` argv claimed the regularizer had been applied.
That is the same silent-no-op class the 2026-09-04 pass fixed for `--ema-decay`
/ `--hard-pair-frac` / `--dro-eta` / `--listwise-weight`, in the other
direction.

**Fixed two ways, both landed:**

1. **The trainer fails loud** — `coarse_decay_unsupported_flag`, written to the
   exact shape of `group_l1_unsupported_flag`, refusing `--coarse-decay` /
   `--coarse-l2-mult` on `--pool-head`, `--hybrid-head`,
   `--per-sample-alpha-head` and `--gpu-runtime`. Two tests, one of which pins
   the **default case as still allowed** (every fast-class and 944-class recipe
   in this repo passes `--coarse-decay 1e-5` on the plain path, so a guard that
   refused it there would break all of them) and one asserting the two riders
   agree on every head reason. Wiring the decay INTO the alpha loop is a real
   optimizer change and is deliberately not done under a guard commit.
2. **The arm design absorbs it.** `train_156_student.sh` DROPS the flag on an
   alpha-head cell with a printed note, and gains `WR4_NO_COARSE_DECAY=1` so the
   plain-path **no-decay control** the alpha arms must be read against can be
   built. Plain-path cells are byte-identical to before (`${CDECAY:+--coarse-decay
   "$CDECAY"}` with `CDECAY=1e-5` expands to exactly `--coarse-decay 1e-5`), so
   gate G1 is unaffected.

**Registered as an ADDITION to Phase C: `P0nd`** — the base recipe on the plain
path with `WR4_NO_COARSE_DECAY=1`, k = 3. Without it, `P1α − control` is not a
head effect. The verifying build was done in a **separate cargo target dir** so
the running sweep's pinned binary was not replaced mid-wave; this wave's fits
therefore predate the guard, and the runner is what keeps them honest.

## 5. THE W4 PLAN, REVISED ON THE KERNEL LANE'S DEFECT 2

`kernel_fastclass_2026-09-05.md` §4 measured that the W4 **bar arm itself** is
unproducible: `ssim2_speed_bar`'s `add156_156basic` builds
`V1PoolsMode::Off`, and `fold_engine::pools_mode_for_need` (`fold_engine.rs:538`)
**never returns `Off`** by documented policy (it hands the band no scratch,
disabling the band-local self-blur shape). *"The bar is therefore set by a walk
that only the bench can run."*

**For this campaign's servable candidate that resolves the denominator
question rather than complicating it.** A 228-slice bake is served through
`V1PoolsMode::Peaks` — the mode `ZensimProfile::D` already resolves to — so its
walk is **the production D walk**, not a bench-only one. The honest ≤1.25× base
is therefore the **`15c` (Peaks) arm**, and the kernel lane's own 1152²/1T
numbers put `156` / `15o` / `15c` at 26.73 / 25.56 / 25.76 ms, all inside that
cell's control spread — so the two candidate denominators are not distinguished
at 1T anyway.

**Revised plan, in two halves:**

1. **Walk cost** — `scripts/kernel_fastclass_sweep.sh --arms 156,15c,15f
   --control 15c --sizes 576,1152,2304 --threads 1,8 --starts 15 --iters 7`.
   That script *is* the protocol (3-char arm names so the env block is
   byte-identical, arms interleaved inside the start loop, min-of-iters
   in-process, min over ≥15 starts, ASLR on, a bit-identical control whose own
   min-vs-median spread IS the cell's noise floor, and a **box-load self-check
   that skips rather than emitting a contaminated number**). It refuses above
   load 3.0, which is why W4 runs only after the fit queue drains.
2. **Forward-pass cost** — `ssim2_speed_bar` with `ZEN_S2_EXTRACT_ONLY=1`, which
   is the only thing that separates "the walk got wider" from "the head got
   bigger". For a servable 228 candidate the walk delta is ~0 by construction,
   so the entire W4 question is the forward pass: a `265→128→1`-class MLP
   against Profile D's 28-coefficient additive head.

**And the baseline is expected to move.** The kernel lane measures the fast
walk at **6.5 ms @576²/1T** (944-full 16.2) with the **front end** — XYB
convert + downscale — at a third of it, and a separate lane is optimising that
now. So W4 is measured **last, on the same binary as its baseline**, and no
number is carried forward from an earlier build.

## 6. GATE G7 — SERVABILITY MEASURED, and a byte-identical short-circuit nobody had connected to C5

### 6.1 The servable lane is not a hypothesis

`zensim/examples/serve_custom_bake.rs` (new) loads an arbitrary ZNPR through
`ZensimProfile::Custom` and calls the **production** `Zensim::compute` on real
pixels — the same way the `d_ship_flip` lane found the 944 refusal. On one
`(q1, q3)` zenjpeg pair:

| bake | declared | `Zensim::compute` |
|---|---|---|
| shipped Profile D | `n_inputs=372 caller=372` | **SERVED** −76.863084 |
| `Dpeaks372_id100negrich_dial` — **reads peaks** | `n_inputs=372 caller=372` | **SERVED** −59.302644 |
| `F2_S265_H128_p_s4004_id100` (this lane's control) | `n_inputs=265 caller=944` | **REFUSED** `ModelForwardFailed` |
| `Fpeaks_id100negrich` (D+free's 944 arm) | `n_inputs=944 caller=944` | **REFUSED** `ModelForwardFailed` |

Both sides of the kernel lane's reading confirmed, and the part that needed
checking — **a 372-layout bake that reads the peaks block serves today** —
holds.

### 6.2 `Zensim::compute` short-circuits byte-identical input BEFORE the model

The refused bakes still printed `IDENTITY (ref vs ref) score=100.000000`. That
is not the error being swallowed: `metric.rs` builds `(100.0, 0.0, zeros)` and
calls `.mark_identical()` (:3509, :5225), and `apply_mlp_scoring_with_codec`
reads the flag to **skip the MLP forward** — with the reason stated in the
field's own doc: the forward *"produces garbage (the bake has no signal
anchoring 'zero feature vector → score 100')"*.

**So for byte-identical input the product returns exactly 100 whatever bake is
loaded, and it does so without consulting the dial at all.** That reframes what
C5 protects, and the reframing is worth stating because the gate's own owner
constant is written in terms of `dial(0⃗)`:

* C5 is a property of the **bake** — what its dial does at the identity feature
  vector — and remains the right gate, because the short-circuit is keyed on
  **byte-identical pixels**, not on near-identity. A one-bit difference, a
  lossless re-encode, a q=100 JPEG: none is byte-identical, all go through the
  model, and *that* is the regime a near-lossless product dial lives in.
* But a C5 failure is **not** a claim that `zensim(x, x) != 100` in production.
  It never is. Reporting it as one would be wrong.

Neither point changes any number in this campaign; both change how the C5 row
should be read, and neither was on the record.

## 7. A PRE-FLIGHT THAT SAVED 15 FITS — the 372 lane's target scale

Before the servable lane ran a single cell, its six training legs were checked
for target ORIENTATION and target SCALE. Orientation was clean —
`corr(human_score, ssim2_gpu)` reads **+0.803** (kadid), **+0.849** (tid),
**+1.000** (safesyn), **+1.000** (cid22_train_norm), all quality-oriented, so
the registered KADID inversion does not touch this root.

**Scale was not.** The recipe convention is `human_score` in **[0,1]** with
`--target-scale 100`, and the 372 directory carries both forms under names that
differ by one suffix:

| leg (372) | measured range | 944 twin's range |
|---|---|---|
| `safesyn.parquet` | [−7.3904, 0.9870] | [−7.3904, 0.9762] |
| `kadid.parquet` | [0.0000, 0.9825] | [0.0000, 0.9825] |
| `tid.parquet` | [0.0269, 0.8016] | [0.0269, 0.8016] |
| `tbig_372_200k.parquet` | [0.0000, 0.9840] | [0.0000, 0.9840] |
| **`cid22_train.parquet`** | **[3.0102, 94.1532]** ✗ | [0.0301, 0.9415] |
| ↳ `cid22_train_norm.parquet` | [0.0301, 0.9415] ✓ | *(identical)* |
| **`konjnd-dense.parquet`** | **[−65.7108, 96.1549]** ✗ | [−0.6493, 0.9615] |
| ↳ `konjnd-dense-norm.parquet` | [0.0000, 1.0000] ✓ | — |

The first draft of `train_372_student.sh` named the un-normalised pair. After
`--target-scale 100` those two groups would have carried targets ~**100×** the
other four legs', so a RankNet/MSE mix would have been dominated by them
outright — and nothing would have crashed. Fixed to the `_norm` variants before
the queue reached stage 2.

**Four of the six legs match their 944 twins' ranges to the digit**, and
`cid22_train_norm` matches `ext_cid22_train201` exactly, which is the strongest
available evidence that the two lanes are running the same recipe on two
layouts. The two that differ are stated in the script's own header: no
`tbig_hf` twin exists at 372, and the konjnd leg is the older 20,160-row
`konjnd-dense-norm` rather than the 8,060-row BPG split.

### 7.1 The 372 recipe SMOKE-TESTED, and the feature-set id works end to end

One epoch, 2,000 pairs, `--keep-features slice_basic156_peaks.txt`,
`--max-features 372`. Two things came out of it that no amount of reading could
have given:

**(a) The recipe loads and scores.** The trainer's own auto-eval hook produced a
verdict at `--regime 372` on all 14 corpora. At *one epoch* it already reads
CID22 **+0.8236**, KADID 0.840, TID 0.835, CSIQ 0.920, LIVE 0.938, KonJND
0.356 — not a result, but proof the six legs, the target scales, the slice and
the eval root all line up.

**(b) The naming machinery identifies the bake correctly, unprompted:**

```
bake_verdict: feature-set — table basic+peaks+masked+iw@w372/v1cur#d16a1091 [... INFERRED]
bake_verdict: feature-set — bake  basic+peaks@w372/unknown#3fb78648
bake_verdict: feature-set note — EraUnknown: era not established (bake unknown,
  table v1cur) — 'we do not know which extractor made this' has the same
  consequence for a published number as 'a different one'
```

`3fb78648` is this campaign's own registered 228 hash, derived from the bake's
bytes with no hint from the invocation — fundamental 3 working end to end. The
`unknown` era was a real gap and is now closed: `canonical-2026-05-21/train` is
registered as a root (`basic+peaks+masked+iw@w372/v1pre`), along with
`basic+peaks@w372/v1pre#3fb78648` and `basic@w372/v1pre#3ffe8670`.

**Note the hash is layout-independent by design.** `basic+peaks@w372/v1pre` and
`basic+peaks@w944/era2r4` share `#3fb78648` because the hash is over the SLOT
LIST; layout is a separate component of the id. They are the same reader set at
two layouts — and that distinction is exactly what decides servability (§6.1),
which is a good argument that the id grammar has the right shape.

The 372 lane therefore trains on **v1pre** and evaluates on **v1cur**. That is
stated rather than hidden; the flip lane's own 372 era A/B bounds the rank skew
at **≤ 7e-4**.

## 8. THE SLICE EFFECT ON A7r, ISOLATED — class and layout held fixed

§3.3 could only say "everything past `f0..155` sits at 4 or 5". This is the
controlled version: **same model class (sparse additive lasso), same layout
(372), same id100+negrich anchor chain, same instrument, one variable — the
slice.** Both bakes come from the d_peaks lane's own arm set.

| bake | slice | contract | **A7r** | C1 mono | avif-rav1e | avif-svt | jpeg | jxl | webp |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| `CTL_d_id100_negrich_dial` | **156** | PASS | **0** | 0.9931 | 0.6667 ✓ | 1.0000 ✓ | 0.6667 ✓ | 1.0000 ✓ | 1.0000 ✓ |
| `Dpeaks372_id100negrich_dial` | **228** (+72 peaks) | PASS | **3** | 0.9616 | **0.2821 ✗** | 1.0000 ✓ | **0.5641 ✗** | **0.6923 ✗** | 1.0000 ✓ |
| *mentor `peer_ssim2`* | — | — | — | — | *0.6410* | *1.0000* | *0.6667* | *0.9615* | *1.0000* |
| *(context)* `v47_strict_qat_native` | 372-full, **MLP** | PASS | 4 | 0.9803 | 0.3590 ✗ | 0.8462 ✗ | 0.5128 ✗ | 0.8462 ✗ | 1.0000 ✓ |
| *(context)* shipped Profile B | 372-full linear | FAIL | 5 | 0.9776 | 0.1795 ✗ | 0.4359 ✗ | 0.5641 ✗ | 0.4231 ✗ | 0.9487 ✗ |

**Adding the 72 peaks costs three of five codecs on floor representability, and
0.0315 of dial monotonicity, at fixed class and fixed layout.** The d_peaks
lane reached the same attribution from a wider comparison; this removes the
remaining confounds (different λ, different arms) and puts a number on it.

**Consequence for this campaign's ship decision, stated before its own arms
land.** The registered ship rule needs A7r PASS. On this evidence:

* **`S156` is the shape most likely to ship** — it is the only slice any scorer
  has ever passed A7r with, in either class.
* **`S228` is likely to fail A7r**, and it is the slice with the rank upside
  (the D+free lane measured the peaks half carrying 97 % of the free set's
  CID22 gain). So the campaign is walking into a **rank-vs-floor trade**, which
  is exactly the trade the d_peaks slot-ablation lane hit from the other side
  (dropping `f162` fixed jxl's A7r and created a new A4 failure — *"a clean
  either/or with no arm failing both"*).
* **Two things are still genuinely open**, and this campaign measures both:
  whether an **MLP** over the same slice inverts the same way (every model in
  the table above is a linear fit except `v47`, which is 372-**full**, not
  228), and whether **`--monotonicity-reg`** — the only ordering-aware
  regularizer in the trainer, alpha-head-only, arm `MR` — moves it.

## 9. FIRST ARM RESULT — the SERVABLE set beats the control on the campaign's own axis

Two Phase-A cells are complete at k = 3 (seeds 4004/4005/4006), both at
H128 on the plain path, both read from their own fullevals:

| arm | k | composite | CID22 | **KonJND** | AIC-3 | CSIQ | LIVE | imazen26 | nonphoto | hfnlproxy | mono | **A7r** | bytes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **`S228`** (156+peaks — **SERVABLE**) | 3 | **0.8652** | 0.8852 | **0.4536** | 0.7985 | 0.9556 | 0.9282 | 0.9487 | 0.9465 | 0.4173 | 0.9901 | 5 | 29,298 |
| `S265` (156+free — the control) | 3 | 0.8645 | **0.8863** | 0.4322 | 0.8018 | 0.9559 | 0.9442 | 0.9504 | 0.9487 | 0.4271 | 0.9901 | 5 | 30,669 |

Per-seed spreads: CID22 0.0041 / 0.0018, KonJND **0.0375 / 0.0146**.

**Three readings, and the first one is the campaign's headline so far.**

1. **Dropping the 37 raw-moment slots — which is what makes the set servable —
   COSTS NOTHING and buys KonJND.** Composite +0.0007, KonJND **+0.0214**,
   CID22 −0.0011 (a quarter of the CID22 spread, and well inside the 0.0069
   per-model CI half-width). That is the same direction the D+free lane
   measured in the *linear* class (peaks carry 97 % of the free set's CID22
   gain, and beat the full free set on 8 of 12 corpora) — **now reproduced in
   the MLP class, where it had never been tested.**
2. **Against the era-closed bar** (composite ≥ 0.8626, CID22 ≥ 0.8877, KonJND ≥
   0.4782): `S228` **passes composite outright** — above both 944 leaders —
   ties CID22 within the CI, and closes the KonJND gap from the incumbent's
   −0.046 to **−0.025**.
3. **A7r is unmoved at 5**, for both. The floor blocker does not care which of
   these two sets it is looking at, which is consistent with §8's isolation:
   the step that costs codecs is 156 → 228, and both of these are past it.

**Instrument note for the endgame.** `freeze_check --select --seed-group
--min-k 2 --floor-basis all` runs and correctly reports **NO SELECTABLE
RECIPE** — both groups are `replicated` at k = 3 with 7-8/8 floors, but M3a is
UNMEASURED, and the owner's registered rule lists an UNMEASURED cell without
making it selectable. M3a therefore has to be measured before selection, via
its owner (`scripts/harvest_bakes.sh`, 27 cells, ~66 s/bake). At ~60 bakes that
is ~66 min of exclusive CPU, so it runs on the curated candidate set after the
fit queue drains — never concurrently, which would corrupt both.

---

# RESULTS

## 10. THE CAPACITY TABLE — 944 lane (era2r4, k=3 per cell, seeds 4004/4005/4006)

| arm | composite | CID22 | KonJND | AIC-3 | CSIQ | LIVE | im26 | nonphoto | hfnl | mono | **A7r** | bytes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `S156` H128 | 0.8400 | 0.8752 | 0.4189 | 0.8098 | 0.9568 | 0.9302 | 0.9021 | 0.8962 | 0.3681 | **0.9951** | 5 | 20,323 |
| `S156` H32 | 0.8421 | 0.8790 | 0.4220 | 0.8078 | 0.9522 | 0.9420 | 0.9025 | 0.8972 | 0.3692 | 0.9948 | 5 | **15,584** |
| `S228` H128 | 0.8652 | 0.8852 | 0.4536 | 0.7985 | 0.9556 | 0.9282 | 0.9487 | 0.9465 | 0.4173 | 0.9901 | 5 | 29,298 |
| **`S228` H32** | **0.8666** | **0.8887** | 0.4543 | 0.7999 | 0.9571 | 0.9516 | 0.9482 | 0.9452 | 0.4168 | 0.9894 | 5 | 20,231 |
| `S261` H128 | 0.8660 | 0.8833 | **0.4566** | 0.8053 | 0.9574 | 0.9404 | 0.9518 | 0.9491 | 0.4336 | 0.9896 | 5 | 35,308 |
| `S261` H32 | 0.8622 | 0.8861 | 0.4156 | 0.8048 | 0.9555 | 0.9402 | 0.9487 | 0.9453 | 0.4181 | 0.9919 | 5 | 20,322 |
| `S265` H128 *(control)* | 0.8645 | 0.8863 | 0.4322 | 0.8018 | 0.9559 | 0.9442 | 0.9504 | 0.9487 | 0.4271 | 0.9901 | 5 | 30,669 |
| `S265` H32 | 0.8640 | 0.8866 | 0.4237 | 0.8003 | 0.9561 | 0.9486 | 0.9517 | 0.9490 | 0.4216 | 0.9903 | 5 | 21,692 |
| `S289` H128 | 0.8633 | 0.8836 | 0.4301 | 0.7999 | 0.9545 | 0.9301 | 0.9516 | 0.9497 | 0.4257 | 0.9896 | 5 | 36,339 |
| `S289` H32 | 0.8614 | 0.8859 | 0.4048 | 0.8031 | 0.9560 | 0.9271 | 0.9497 | 0.9468 | 0.4190 | 0.9904 | 5 | 21,444 |
| `S265` H128 **+skip** | 0.8645 | 0.8863 | 0.4322 | 0.8018 | 0.9559 | 0.9442 | 0.9504 | 0.9487 | 0.4271 | 0.9901 | 5 | 30,696 |
| `S265` H128 **no-decay** | 0.8635 | 0.8844 | 0.4267 | 0.8014 | 0.9581 | 0.9327 | 0.9516 | 0.9501 | 0.4187 | 0.9908 | 5 | 32,788 |
| `S265` H128 **α-head** | **0.5710** | **−0.5860** | 0.1119 | −0.6231 | −0.0940 | −0.1953 | −0.6564 | −0.6707 | −0.2808 | 0.9606 | 5 | 68,961 |
| **`SORACLE`** (944-full) | **0.8581** | 0.8831 | 0.4191 | 0.7960 | 0.9620 | **0.9669** | 0.9414 | 0.9385 | 0.4214 | 0.9943 | 5 | 74,327 |

**Five things this table settles.**

1. **CAPACITY IS NOT A LEVER.** H32 vs H128 moves composite by −0.0038…+0.0021
   across six set×width pairs — inside every per-seed spread — while cutting the
   bake **30–47 %** (`S156` 20,323 → 15,584 B; `S228` 29,298 → 20,231 B). The
   registered H256 extension is therefore **NOT RUN**: a width axis that is flat
   downward and costs bytes upward has answered its own question, and spending
   six fits to confirm it would have been the sunk-cost move.
2. **THE COMPUTE CEILING IS *BELOW* THE RESTRICTED SETS.** `SORACLE` — the same
   recipe with no `--keep-features`, free to read all 944 coordinates — reads
   composite **0.8581**, the *lowest* non-degenerate cell in the table, and
   KonJND 0.4191 against `S228`'s 0.4543. **The fast class's KonJND gap is NOT a
   compute gap.** That was this campaign's standing hypothesis (Phase A-ORACLE
   was registered to test it) and it is falsified: giving the recipe every
   feature makes it *worse*.
3. **THE SERVING-PATH MARGIN IS NOT MET — decided by its pre-registered rule.**
   Best of {S261, S265, S289} = `S261` H128 at composite 0.8660 / CID22 0.8833,
   against `S228` H32's 0.8666 / 0.8887: **−0.0006 composite and −0.0054 CID22,
   both NEGATIVE.** The rule required ≥ +0.0070 or ≥ +0.0069. **Building the
   944-layout scoring path is not justified**, and the un-servable sets are
   closed as a direction.
4. **The per-sample-α head is a catastrophic regression on this recipe** —
   CID22 **−0.5860**, i.e. an inverted ranker — and its 2-layer sibling could
   not even be packed (`packed tanh-pin range [−38.4, 14.2] corr=−0.8350`;
   `error: packed-network spline fit produced only 1 knots (<2)`, all 3 seeds).
   So **arms C2/C3 do NOT run**: their frozen precondition was that `P1α` not be
   a KonJND regression, and it is a regression on every axis. `--monotonicity-reg`
   stays UNMEASURED with a named cause, never reported as a null.
5. **`--skip-connection` is byte-for-byte inert on rank** (every axis identical
   to 4 dp against the control, bakes differing by 27 B) and **`--coarse-decay`
   is worth +0.0010 composite / +0.0019 CID22 / +0.0055 KonJND** — small, real,
   and exactly why the `nd` control had to exist.

**A7r is 5 in every cell, including `S156` and `SORACLE`.** At 944 layout the
floor gate does not discriminate between slices at all — which, against §8's
372 result (156 → 228 costs three codecs there), says the 944 MLP class fails it
for a reason of its own.
