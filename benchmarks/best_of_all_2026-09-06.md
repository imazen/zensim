# BEST-OF-ALL — a CONSTRAINED MLP on the 228 servable slots (2026-09-06)

Plan (pre-registered before any code): [`docs/PLAN_BEST_OF_ALL_2026-09-06.md`](../docs/PLAN_BEST_OF_ALL_2026-09-06.md).
Artifacts: `/mnt/v/output/zensim/best-of-all-2026-09-06/`.
Lane: sibling jj workspace `zensim--bestofall`.

**Thesis.** `benchmarks/fastclass2_campaign_2026-09-05.md` measured a 228-input
MLP that ranks with the 944 leaders (CID22 0.889636 at k=3, +0.0263 over shipped
D) and is faster than shipped D in every W4 cell — and that fails the dial
contract on **identity (90.9368, 38 of 38 probe rows outside the band)** with
**1,642 of 9,593 grid cells above identity (17.1 %)** and **A7r failing on 5 of
5 codecs**. `benchmarks/dial_addressability_gate_2026-09-04.md` §10.3 proves the
matching either/or on the shipped-B lineage: **no monotone output spline can
satisfy both C2 and C6** when real cells out-rank a perfect copy in raw space.
So the dial cannot be repaired downstream of the weights. This lane makes the
identity and no-cell-above-identity rows **structural**.

---

## 1. PHASE A — four owner defects, all read from source, all one family

The trainer had **a convention with no owner**. Four instances, all landed with
a failing-first gate and a byte-identity control.

### 1.1 Output polarity was applied at 1 of 8 polarity-sensitive sites

`mlp_train/mod.rs:2088` derives `rank_target_sign` (`−1.0` when any group
carries an absolute/MSE term, else `+1.0`) with a correct comment and a
2026-07-15 measurement behind it — HF held-out per-ref SROCC **+0.6393 →
−0.3454** when rank supervision was *added*, i.e. adding supervision made a
corpus rank backwards. It was used at **one** site (`:2556`). Every other site
carried its own, opposite, hard-coded assumption:

| site | what it assumed | reached by |
|---|---|---|
| `train_mlp_strategy` sequential RankNet | reconciled | plain path, `--minibatch-size 1` |
| `train_mlp_pool_head_with_tv` | bare `signum` | `--pool-head` |
| `train_mlp_hybrid_head_with_tv` | bare `signum` | `--hybrid-head` |
| `run_parallel_minibatch` | bare `signum` | **plain path**, `--minibatch-size > 1` |
| `run_minibatch_with_nin` | bare `signum` | **plain path**, `--norm-in-norm-weight > 0` |
| `train_mlp_per_sample_alpha_head` | bare `signum` | the only depth-2 / skip path |
| the TV within-ladder hinge (×2) | DISTANCE, in prose | every path |
| the α head's monotonicity hinge | SCORE | the α head — *against its own RankNet term* |

**MEASURED — the campaign's α-head defect reproduced in a 0.5-second unit test.**
On a synthetic corpus, an α-head recipe carrying an absolute term trains a model
whose ordering is perfect and whose sign is backwards: raw SROCC **−0.9970** at
depth 1 and **−0.9986** at depth 2, where the plain path on the same data reads
**+0.99**. That is the same shape as the campaign's measured raw CID22
**−0.8921** — the campaign's single best CID22 *ordering*, arriving negative,
which `bake_dial_refit pack` then cannot spline at all because the output
calibration spline is monotone increasing by construction (all three seeds died
with `HARVEST_FAILED`).

**Mechanism, confirmed.** The winning 228 recipe declares `:both` on 4 of its 6
legs, so `rank_target_sign = −1.0` on the plain path (SCORE-shaped) while the α
head stayed on the legacy DISTANCE convention. Two terms, one output, opposite
directions.

**Fix.** `OutputPolarity::{Distance, Score}`, derived once by `for_groups` and
asked by every site. `rank_target_sign()` for the RankNet term; `ladder_sign()`
for both ordering hinges, which now share the single form
`max(0, ladder_sign·(y_better − y_worse) + margin)`. The monotonicity hinge
reads a new `quality_sign` — *which member is better is a property of the human
scores, never of the output convention*. `Distance` is the default, so every
rank-only recipe is byte-identical, and every path now logs the convention it
trained under.

**Negative control (the important half).** The test file imports
`OutputPolarity`, a type this change introduces, so it does not compile at the
parent — the control was run as a **stripped variant** with that import and the
owner-table test removed, leaving five tests. Of those five, **4 FAIL at
`main@origin` `0c6307a7`**. The 5th — "rank-only recipes are DISTANCE-shaped on
every path" — passes there and here, which is the point. *(The shipped file's own
header said "Five of these arms fail"; corrected to 4 of 5.)*

### 1.2 `run_parallel_minibatch` / `run_minibatch_with_nin` silently dropped the absolute term

Both are pure RankNet + PWRC. A run that set an absolute term *and* `K>1` (or
NiN) threw the absolute term away without a word. They now **refuse** — this
dispatcher's established response to the class (it already fails loud for seven
sibling flags). Rank-only recipes at any `K` are untouched.

### 1.3 `--n-hidden-layers` and `--skip-connection` were silent no-ops off the α-head path

`use_2layer` / `use_skip` are read at exactly two lines, both inside
`train_mlp_per_sample_alpha_head`. A run that asked for depth 2 or a skip
connection on any other head got a 1-layer, skip-less net and a bake
byte-identical to one that never asked. Now refused.

### 1.4 `--leaky-alpha` was a train/serve divergence

Every bake emitter hard-coded `Activation::LeakyRelu`, and the runtime applies a
hard-coded `zenpredict::LEAKY_RELU_ALPHA` for that byte. So any run with a
different slope trained one function and served another, silently.
`Activation::Relu` (byte 1) has been in the wire format and the runtime all
along and was simply unreachable from this trainer. `hidden_activation_for` now
derives the byte from the slope and **panics** on anything unrepresentable.

**Found while implementing it:** the runtime constant is an **f32**, so
`LEAKY_RELU_ALPHA as f64 = 0.009999999776482582` — *not* the f64 literal `0.01`
the trainer defaults to. Both are accepted; that ~2.2e-10 relative gap is the
(harmless, but real) slope divergence every LeakyReLU bake has always carried.

### 1.5 Byte-identity control

`tests/legacy_bake_sha.rs` pins five rank-only recipes to sha256 digests
MEASURED at `main@origin` `0c6307a7`. All five reproduce byte-for-byte after all
four fixes, **including both TV arms** — which is what proves the
newly-reachable `--tv-margin` is the identical function at its `0.0` default.

| recipe | sha256 | bytes |
|---|---|--:|
| `PLAIN_RANK_ONLY` | `ca172d6339d8b7b5…` | 996 |
| `PLAIN_RANK_ONLY_TV` | `b57f7edf5024a783…` | 996 |
| `PLAIN_RANK_K32` | `e41d39daaeda67d8…` | 996 |
| `ALPHA_RANK_ONLY` | `691291aae24bfb25…` | 1,732 |
| `ALPHA_RANK_ONLY_TV` | `561dccb5bdd2cfb1…` | 1,732 |

Blast radius on published work: **none**. The α-head arms this fixes have no
fulleval on the board, and no board bake sets `--minibatch-size > 1` or NiN with
an absolute term.

---

## 2. THE ARCHITECTURE — `--nonneg-distance`

Pre-registered choice, and it is **not** one of the three the brief offered.
softplus / ReLU² / squared-norm each need a new `Activation` or a new head in
`zenpredict` — a wire-format change plus a change in every serving consumer —
for a property this form already gets exactly, and `softplus(0) = ln 2` needs a
constant offset baked somewhere, which is one more place for the pin to drift.

> `raw(x) = pin − g(x)`, realized as a ReLU encoder with zeroed hidden biases
> and a sign-constrained final layer **whose bias IS the pin**:
>
> - `scaler_mean := 0⃗` (scale-only standardization) — the identity feature
>   vector is MEASURED to be exactly `0⃗` on all 372 slots for every image, and
>   subtracting a mean would map it to `−μ/σ ≠ 0⃗`
> - hidden biases frozen at `0.0` ⇒ `h(0⃗) = ReLU(0⃗) = 0⃗`
> - hidden activation **ReLU** ⇒ `h ≥ 0` for every input
> - output weights projected `≤ 0` ⇒ `raw = w₂·h + pin ≤ pin`, always
> - output bias frozen at `pin` ⇒ `raw(0⃗) = pin`, bit-exactly

**`raw(0⃗)` is the argmax of `raw` over the entire input space, by
construction.** The pin lives in the output bias precisely so this is expressible
in the SHIPPED wire format and the SHIPPED runtime with zero changes to either,
and so the absolute term — which regresses raw output onto a target in `[0,100]`
— stays satisfiable.

### 2.1 Gated, not asserted

`tests/nonneg_distance.rs`, 8 tests:

- `raw(0⃗) == pin` by **`to_bits()` equality** at **F32, F16 and I8** — not a
  tolerance. `0·w = 0` exactly at every dtype, `ReLU(0) = 0`, `dot(0⃗, w) = 0`,
  and the frozen bias passes through. If this ever needs a tolerance the
  guarantee is gone. Also gated at a non-default pin, so the constant is data
  rather than a magic number.
- `raw(x) ≤ pin` over **120,000 probes** spanning six magnitude scales from
  `1e-30` to `1e12` and both signs — far outside anything the fit saw, because
  C6 is a claim about the whole input space, not about the eval grid.
- **the pin is ATTAINED** (some probe lands exactly on it) — the bound is tight,
  which is what makes `raw(0⃗)` the argmax rather than merely an upper bound.
  *(This corrected the test, not the code: the first draft asserted the max was
  strictly below the pin, which is wrong — an input that turns every hidden unit
  off legitimately lands on it.)*
- `--leaky-alpha 0` bakes ReLU, the default bakes LeakyRelu, an unrepresentable
  slope panics.
- All three refusals (`--skip-connection`, any head flag, `--n-hidden-layers ≥ 2`).

The non-negativity tests double as the **ReLU train/serve gate**: the guarantee
depends on `h ≥ 0`, so a bake declaring `LeakyRelu` while the fit used ReLU
would let `h` go negative, `w₂·h` go positive, and `raw` exceed the pin.

### 2.2 What is NOT claimed

C3/C4 (the negative floor) and A7r (per-codec ordering at the bottom of ladders)
do **not** follow from the architecture: the first is a spline-anchor question,
the second a ladder-supervision one. **Three rows from one mechanism, not six.**

And at one hidden layer `g` is a non-negative combination of `ReLU(linear)`,
i.e. **CONVEX** in the standardized features. That is a real restriction, and
the plain path is 1-hidden-layer only. It is documented on the flag rather than
left to be rediscovered from a rank number, and the wave measures what it costs.

---

## 2.4 ⛔ CORRECTION — "by construction" is CONDITIONAL, and this wave does not meet the condition

**Found by adversarial review 2026-09-06, then MEASURED. Every "by
construction" claim above about *identity* is narrower than it was written.**

`--nonneg-distance` establishes its guarantee in **standardized** space. The
chain is `caller row → feature transforms → scaler → layer 0`, and the scaler is
the only step the flag controls. So `raw(identity) = pin` requires that **every
active feature transform maps 0 to 0** — and `winsor_p99` with `lo > 0` does
not: it returns `lo`.

**The canonical 372 transform screen this wave uses carries 28 such guards**
(`winsor_p99:100:1.46128e-06,…`, `:134:0.000331953,…`, `:95:0.000610845,…`, …),
streamed verbatim into every arm. MEASURED on `B_nonneg_s4004` by forwarding the
pinned 38-row all-zero identity probe through the packed bake:

```
B_nonneg_s4004   raw(identity) = 99.61380004882813     (pin = 100.0)
B_nonneg_s4005   raw(identity) = 99.564697265625
A_plain_s4004    raw(identity) = 17.807016372680664    (no constraint — for scale)
```

**What survives, exactly:**

| claim | status |
|---|---|
| `raw(x) ≤ pin` for **every** input | **STILL STRUCTURAL** — the `w₂ ≤ 0` projection with `h ≥ 0` gives it, and it does not depend on the transforms |
| `raw(identity) = pin` bit-exactly | **NOT structural here.** Identity sits **0.386 below** the ceiling |
| identity is the **argmax** | **NOT structural here.** An input that turns every hidden unit off would reach the pin and out-score identity |
| **C5** (identity dial in band) | **still passes, and for a good reason** — the identity ANCHOR rows go through the *same* forward, so `fit_spline_knots` maps that same raw to exactly 100. The dial pin is intact |
| **C6** (no cell above identity) | **passes as a MEASUREMENT on 9,593 real grid cells across three seeds — not as a theorem** |

So the §5.3/§5.5 result stands as measured, and the mechanism claim does not
stand as *proved* for this configuration. C6 = 0 on every seed is strong
evidence that no real codec output reaches the pin; it is not the same statement
as "no input can".

**Landed with the correction:** the trainer now emits a loud warning naming the
count of non-zero-preserving transforms, the worst `t(0)`, and the exact
consequence — so the next run cannot make this claim without seeing the caveat.
It is a **warning, not a refusal**: the property the gate grades was measured to
hold with these transforms active, and refusing would have invalidated a wave
whose answer is good.

**To make it structural**, one of: drop the positive-`lo` winsor guards under
`--nonneg-distance`; re-screen the transforms as zero-preserving (`lo = 0`); or
pin the architecture at `t(0⃗)` instead of `0⃗`. All three are real work and
none was done here.

*(This is also why the plan's `identity_rows_are_a_no_op_under_nonneg_distance`
gate was never written: under these transforms the premise is FALSE — the
identity rows would carry a real, non-zero residual. See §5b D-4.)*

---

## 3. THE LADDER LOSS — the owner already existed

`TvRegularizer` **is** a within-ladder pairwise hinge over adjacent severity
levels with an anti-collapse margin, wired on the plain path, and the distill
wave names it as the untried W3 lever. Adding a `--ladder-hinge` flag would be a
duplicate implementation, which this repo bans; the rule outranks a flag name in
a brief. So the loss is `--tv-weight` / `--tv-margin` / `--tv-band-weights`, and
the new artefacts are **two owner extensions** plus **a builder**.

**Extensions.** `--tv-margin` was α-head-only; the plain path's hinge was a pure
`max(0, y_hi − y_lo)`, and a pure hinge is minimized by collapsing every ladder
flat — flat ladders are `tied`, which is **C2**. And the hinge was
polarity-blind while this recipe is score-shaped, so *the un-flipped hinge would
have trained the ladder backwards*: §1.1 is a hard prerequisite for the ladder
arm, not an independent cleanup.

**Builder** — `scripts/canonical_corpus/build_ladder_tv_pairs.py`, MEASURED:

| | |
|---|--:|
| positional join VERIFIED (every row, not a sample) | **196,086 / 196,086** |
| ladders (`ref` × codec, 6 codecs, 16-step q grid) | 19,259 |
| adjacent pairs considered | 176,825 |
| ladders too short after saturation dedup | 89 |
| dropped below the materiality margin | 1,089 |
| **pairs emitted** | **175,736** |
| — low-q band (`q ≤ 50`) / mid / high | 79,533 / 46,699 / 49,504 |

**Materiality**: a pair is kept only when the reference metric orders its two
members by **≥ 0.5 ssim2 points** — the same `ENCODER_SSIM2_MARGIN_PT` /
`MATERIAL_INV_PT` constant `bake_verdict` grades the dial with, so the hinge is
supervised on exactly the moves the exam calls material and on no others.
Band ids carry the measured low-q concentration (`q<50` holds 28 of 57 material
inversions at 0.03526/pair against 0.00727 and 0.00793 elsewhere).

**Saturation dedup on `size_bytes`, declared as a PROXY** — the ladder
instrument keys on `encode_sha`, that column does not exist in this sidecar, and
nothing has verified that q5 and q10 are distinct settings on every codec.
MEASURED: it collapsed **2 steps out of 196,086 rows**. The plateau worry that
motivated it — zenjpeg emitting one bitstream for all of q 0..10, which is what
made the ladder instrument necessary in the first place — barely materializes on
*this* grid, because safesyn starts at q5 and steps by 5. The check stays,
because "it did not fire on this corpus" is not "it cannot fire", and 2 is not 0.

Index range verified against the group it indexes: **max index 196,085 < 196,086
rows**, i.e. zero out-of-range pairs — which is also what the loader's new
loud-drop counter will confirm at train time rather than assume.

**The two-reference agreement arm is NOT MEASURED, not silently skipped.**
Butteraugli is present on 196,086/196,086 rows but the columns do not name their
variant, and the gate's rule needs pnorm3 (margin 0.05) or max (0.25) *by name*.
Identifying it empirically is what would unlock a second arm.

**One more silent drop closed**: the TV loader ignored out-of-range pairs with a
bare `continue`. It now counts them, says so, and refuses when every pair was
dropped — indices are into the concatenated group rows in `--group` order, so a
mismatch would otherwise produce a silently-decimated ladder term.

---

## 4. THE REV2 AXIS WAS DROPPED, AND WHY (a deviation from the brief, with the numbers)

The brief's arm table crossed `{rev1, rev2}`. It is dropped, on five measured
grounds from `benchmarks/rev2_refit_2026-09-06.md`:

1. **On the fast class the revision is INERT.** Every Δ is smaller than the seed
   spread of the arm it is measured in (CID22 rev1 0.88885 spread 0.00600 vs
   rev2 0.88854 spread 0.00251, Δ **−0.00031**; the largest Δ, AIC-3 −0.00229,
   sits inside a 0.00745 range).
2. **The seed-matched paired bootstrap SIGN-FLIPS** across the three seeds with
   all three CIs excluding zero (−0.00113 / **+0.00240** / −0.00215). The effect
   is not even consistently signed.
3. **Only 3 of 6 groups are substitutable** — `cid22_train`, `bigcodec` and
   `konjnd-dense` have no rev2 extraction — so the arm is **42.6 % in-era by
   trainer group weight**. That is not an A/B.
4. **The transform screen is era-bound.** 10 of the 40 `--feature-transform`
   entries are `winsor_p99` guards sitting on F17 slots whose p99 collapses
   **~4.4×** under rev2, so a rev1-fitted clamp would be applied to a different
   distribution. `screen-transforms` at rev2 is REGISTERED, NOT RUN.
5. **The two rev2 safesyn tables differ on 51.2 % of cells** (37,379,073 of
   72,943,992) because of an **AVIF decoder-era** difference, and *which is right
   is not decided* — under an explicit AVIF backend-rewrite HOLD.

Spending half the wave on a known-inert, structurally-confounded axis whose
input era is unadjudicated is exactly what the DEAD-list discipline is for. The
freed budget went to an arm the brief did not have — **`E_plainlad`, the control
plus the ladder hinge with no architecture change** — because without it a win by
the constrained+ladder arm cannot be attributed to either half.

---

## 5. THE WAVE

Six arms × three seeds (4004 / 4005 / 4006) = **18 fits**, serialized locally
under `run-heavy --mem 16G --jobs 8`, no paid cloud. Binaries frozen to
`/mnt/v/output/zensim/best-of-all-2026-09-06/bin/` with `BINARIES.sha256` before
launch, so a mid-wave rebuild cannot split the toolchain.

| arm | what |
|---|---|
| `A_plain` | **CONTROL** — the fastclass2 winner's recipe, unchanged |
| `B_nonneg` | `+ --nonneg-distance` (architecture only) |
| `C_lad05` | `+ --nonneg-distance` + ladder hinge @ `--tv-weight 0.5` |
| `D_lad20` | `+ --nonneg-distance` + ladder hinge @ `--tv-weight 2.0` |
| `E_plainlad` | control + ladder hinge @ 0.5 — **isolates the LOSS from the ARCHITECTURE** |
| `F_nonneg32` | `B` at `--hidden 32` (bytes/speed variant) |

Every cell runs the identical chain: train → `pack --neg-tail` against a
**merged anchor** → `densify` (the only writer of `zentrain.feature_ids`) →
`bake_verdict` rank on the **postC** root → G-ADDR on the **floor-dense ladder
instrument** with the pnorm3 reference truth,
`--floor-rule resolvable --floor-margin 0.5`.

**The anchor is the negrich dial anchor concatenated with the 21-row identity
anchor** (`instruments/negrich_plus_identity21_anchor.parquet`, 2,021 rows,
sha256 `2cc9be2edb44016a…`; the 2,000 multiband rows keep their **147 genuinely
negative** `ssim2_gpu` values down to −64.16, and the 21 identity rows carry
exactly 100.0). That gives `fit_spline_knots` a knot at `(raw(identity), 100)`
in the same pass that quantizes and prunes, so QUANTIZE-then-CALIBRATE is
preserved. `21 / 2,021 = 1.04 %` owns the `≥ p99` top bin exactly as the id100
lane sized it (`n = 38` spills into the next bin and displaces the top real
knot).

*(`shared-anchor` is the more elegant home for a second anchor — it takes
`--anchor` repeatably — but it asserts a SINGLE-LAYER linear bake and these are
`228 → H → 1` MLPs. MEASURED the hard way: the first wave attempt died there
with `expects a single-layer linear bake (got 2 layers)`. Merging the anchors up
front is the same fit.)*

**The control gets the identical chain.** Its `raw(0⃗)` is not the argmax, and
what that costs it is the measurement, not a handicap.

**Every arm's feature-set id is `basic+peaks/v1pre#3fb78648`** — the id already
registered for this compute set — and every arm carries the same, COMMON-MODE
era note against the eval table:

```
feature-set — bake   basic+peaks/v1pre#3fb78648
feature-set — table  basic+peaks+masked+iw@w372/v1postc#d16a1091
feature-set MISMATCH — EraDiffers: bake trained on era v1pre, table is era
  v1postc — the shift is model-specific, so the number cannot be corrected
  across the boundary, only re-verdicted
```

That is train-on-`canonical-2026-05-21` / eval-on-`postC`, which is the
campaign's own split and is stated rather than hidden (the flip lane bounds the
372 era rank skew at ≤ 7e-4). It applies **identically to the control and every
arm**, so it cannot move a within-wave comparison — but it does mean none of
these absolute numbers is directly comparable to a bake trained in the postC
era.

### 5.1 Two structural facts the wave had to be rebuilt around, both MEASURED

**(a) `shared-anchor` is single-layer-only.** The first attempt chained
`pack` → `shared-anchor` (which takes `--anchor` repeatably, and is where the
21-row identity anchor naturally belongs) → `densify`. It died with
`bake_dial_refit expects a single-layer linear bake (got 2 layers)`. Merging the
two anchors into one parquet up front is the same fit in one pass, and keeps
QUANTIZE-then-CALIBRATE.

**(b) A DENSIFIED bake cannot be graded by the pinned probes, so the scored bake
is the PACKED one.** `densify` on a contiguous-prefix read set (`f0..f227`)
collapses the caller width **372 → 228** and declares an IDENTITY layout. Every
registered negtail/identity probe is **372-wide**, and `bake_verdict` scores a
probe only when its column count equals the bake's `caller_input_width` — so the
densified bake read **C3, C4, C5 and C6 all NOT MEASURED**, i.e. the entire
contract tier, and the gaddr headline came back
`contract INCOMPLETE (not a pass)`. That is the whole point of this lane, so the
scoring path uses `_packed.bin`, which keeps the 372 caller width via 144 `Drop`
transforms — exactly the shape of the campaign's own published 228 bakes.
`densify` still runs, as a **servability artifact** whose failure is reported and
not fatal.

*(The same investigation found and fixed a real `densify` defect on the way: its
identity gate fed both arms the pre-densify caller-width row, which a
contiguous-prefix read set cannot accept. Every shipped bake it had been run on
reads a SCATTERED id set, so the path had never been exercised. Gated with the
scattered control beside it.)*

### 5.2 The control reproduces the published campaign

First cell, `A_plain_s4004`, against `benchmarks/fastclass2_campaign_2026-09-05.md`:

| | this lane | campaign (s4004) |
|---|--:|--:|
| CID22 | 0.8904 | 0.890817 |
| composite | 0.8728 | 0.873119 |
| KonJND \|·\| | 0.4947 | 0.495731 |
| A7r | 5 / 5 fail | 5 / 5 fail |
| floors (rav1e / svt / jpeg / jxl / webp) | 0.1795 / 0.8205 / 0.5641 / 0.3846 / 0.9744 | identical |

The small rank deltas are the expected consequence of a different pack anchor
(negrich + identity, unclamped `ssim2_gpu`) against the campaign's clamped
`target_score`. **The per-codec floors are bit-identical**, which is what says
the instrument and the ruler are the same ones.

At **k = 3** the control reads CID22 **0.8891 ±0.0042**, KonJND **0.4997
±0.0117**, composite **0.8729 ±0.0008** — against the campaign's published k=3
means of 0.889636 / 0.499906 / 0.873156. Contract **4/6** on all three seeds
(C5 and C6 fail), A7r **5 of 5 fail**, identity 38/38 outside the band, a mean of
**1,491** cells above identity.

**One thing my chain fixes that the campaign's did not:** C3 and C4 **PASS** on
the control here (`frac<0` 0.5645, deepest probe −146.04) where the campaign's
`_id100` cell read 0.3985 / −70.34. That is the negrich (unclamped `ssim2_gpu`)
anchor doing its job, and it isolates the remaining failures to exactly the two
rows the architecture targets plus A7r.

**Servability** (`bake_block_profile` on the packed control): layer-0 reads
**228** caller lines — all 156 of `f0..f155` plus **72** of `f156..f371`, with
the other 144 exactly zero. The 72 are `f156..f227`, the peaks, which
`V1PoolsMode::Peaks` — the mode `ZensimProfile::D` already resolves to —
populates. So the walk is the production D walk and the W4 question is
structural rather than statistical. *(Note the `uses_f156_371: true` flag: it is
TRUE here and is NOT the `--regime 944` mis-scoring hazard, which is about the
masked/IW block `f228..f371` — all 144 of those columns are exactly zero.)*

### 5.3 First constrained cell — the architecture delivers, and it delivers exactly what it claimed

`B_nonneg_s4004`, same chain, same instruments, same seed as the control above:

| row | control `A_plain_s4004` | constrained `B_nonneg_s4004` |
|---|--:|--:|
| **C1** monotonicity ≥ 0.93 | 0.94868 PASS | **0.93040 PASS** (thin) |
| **C2** tied ≤ 0.05 | 0.0017 PASS | **0.0000 PASS** |
| **C3** frac < 0 > 0 | 0.5645 PASS | 0.4540 PASS |
| **C4** deepest probe < 0 | −146.04 PASS | **−207.42 PASS** |
| **C5** identity outside band | **38 FAIL** | **0 PASS** |
| **C6** cells above identity | **1,642 FAIL** | **0 PASS** |
| headline | contract **FAIL** | **contract PASS (6/6)** |
| A7r | 5 of 5 fail | 5 of 5 fail |
| grid max | 94.018 | **100.000** |
| CID22 | 0.89036 | 0.88161 |

**Three things this is evidence for, and one it is not.**

1. **C5 and C6 are structural, on real data, through the real chain.** The grid
   max is **exactly 100.000** — the pin survives training, f16 packing,
   dead-column pruning and the spline refit, which is the empirical form of the
   `nonneg_distance_holds_after_pack_and_prune` claim the unit tests could not
   reach.
2. **The C2 ⊻ C6 either/or is DISSOLVED, not traded.** The gate record's proof
   says a spline can buy C6 only by capping cells and paying in `tied`. Here C6
   goes 1,642 → 0 **while `tied` goes 0.0017 → 0.0000**. Nothing was capped;
   the raw ordering changed.
3. **The floor got DEEPER, not shallower** (−146.04 → −207.42), so the
   architecture does not cost the negative tail the negrich anchor buys.
4. **It does NOT fix A7r** — 5 of 5 codecs still fail, which is what §2.2
   pre-registered ("A7r is an ORDERING failure at the bottom of ladders … the
   architecture alone is not" the mechanism). The per-codec floors move in
   *both* directions (avif-rav1e 0.1795 → 0.2051 and jxl 0.3846 → 0.4231 up;
   avif-svt 0.8205 → 0.6923, jpeg 0.5641 → 0.3590, webp 0.9744 → 0.7692 down),
   which is a redistribution, not a fix. The ladder arms are what test that.

**Two honest cautions.** C1 lands at **0.93040** against a 0.93 bar — a pass with
0.0004 of room, and the control had 0.019. And CID22 falls **0.89036 → 0.88161**
at this seed. Both are single-seed readings; the k=3 numbers are what count.

### 5.4 The instrument is validated against the shipped reference, end to end

Shipped D put through the **identical** invocation this wave's arms use — same
packed-bake path, same postC root, same floor-dense ladder grid, same pnorm3
reference truth, same postC probes:

```
shipped D :: SHIPPABLE (regression PASS + contract PASS)
  C1=pass(0.9946870683243013)  C2=pass(0.0)  C3=pass(0.9145)
  C4=pass(-213.14861297607422) C5=pass(0.0)  C6=pass(0.0)  A7r=pass(0.0)
  floors: rav1e 0.6667 · svt 1.0 · jpeg 0.6667 · jxl 1.0 · webp 1.0
  inversion mono_agree=0.99469 single=0.99309 enc_attr=15 unknown=0
```

Every one of those reproduces `benchmarks/rev2_d_arms_2026-09-06.md` §12.3 and
the mentor bars in `benchmarks/dial_addressability_floor_2026-09-04.json`. So the
gate readings in this document are the same readings the shipped record was made
with — verified rather than assumed, which is the only reason a candidate's
numbers here mean anything.

**The two-reference inversion bars, for reading the arms against:**

| scorer | `mono_agree` | dial-attributed rate | encoder-attributed rungs |
|---|--:|--:|--:|
| shipped D | **0.99469** | 0.53 % | 15 |
| `peer_ssim2` (the mentor) | 0.99160 | 0.84 % | 26 |
| this lane's control `A_plain` (k=3) | 0.94868 / 0.94506 / 0.94432 | ~5.4 % | 15 / 16 / 15 |
| `B_nonneg_s4004` | 0.93040 | 6.96 % | 13 |

C1's bar is 0.93, so both pass it — but both are an order of magnitude worse
than the shipped dial on the axis C1 measures, and **the architecture makes it
worse, not better** (0.94868 → 0.93040 at the matched seed). That is the gap the
ladder arms exist to close, and it is the honest reading of the cost so far.

### 5.5 The constrained arm at k = 3 — the contract is bought, and here is the price

| arm | k | CID22 | KonJND \|·\| | AIC-3 | composite | contract | A7r | identity outside | above-identity |
|---|--:|--:|--:|--:|--:|:--:|--:|--:|--:|
| `A_plain` (control) | 3 | 0.8891 ±0.0042 | 0.4997 ±0.0117 | 0.7974 ±0.0059 | 0.8729 ±0.0008 | **4/6** | 5.0 | 38.0 | 1491.3 |
| `B_nonneg` | 3 | 0.8800 ±0.0049 | 0.4987 ±0.0251 | 0.7878 ±0.0072 | 0.8646 ±0.0020 | **6/6** | 5.0 | **0.0** | **0.0** |
| shipped D | — | 0.86333 | 0.53670 | 0.77700 | 0.82444 | **6/6** | **0.0** | 0 | 0 |

**CONTRACT 6/6 ON ALL THREE SEEDS.** `identity outside` and `above-identity` are
`0.0` as *means over three seeds*, i.e. every seed, not a lucky draw — which is
what "by construction" is supposed to mean and is the point of measuring it k=3.

**THE CONSTRAINT COST, at matched seeds, capacity and features:**

| axis | Δ (`B_nonneg` − `A_plain`) | control's own seed spread |
|---|--:|--:|
| CID22 | **−0.0091** | 0.0042 |
| AIC-3 | **−0.0096** | 0.0059 |
| composite | **−0.0083** | 0.0008 |
| KonJND \|·\| | −0.0010 | 0.0117 |

CID22, AIC-3 and composite all move by **more than the control's seed spread**,
so the cost is real and not a draw. KonJND is a wash inside its own (large)
spread. This is the number the plan pre-registered as "the finding either way",
and it is: **contract 6/6 costs about 0.009 CID22.**

For scale: `B_nonneg` still sits **+0.0167 CID22 over shipped D** while matching
its contract. The blocker is not rank.

**A7r is the blocker, and the architecture makes the floors WORSE.**

| | rav1e | svt | jpeg | jxl | webp |
|---|--:|--:|--:|--:|--:|
| mentor bar | 0.6410 | 1.0000 | 0.6667 | 0.9615 | 1.0000 |
| `A_plain` (k=3 mean) | 0.1538 | 0.7778 | 0.5470 | 0.4231 | 0.9658 |
| `B_nonneg` (k=3 mean) | **0.1880** | 0.6068 | 0.4017 | 0.3205 | 0.8034 |

One codec up, four down. §2.2 pre-registered that the architecture does not
address A7r; this is stronger than "does not help" — the constraint **costs**
floor representability on four of five codecs. Whatever the ladder arms do, that
is the baseline they have to climb out of.

### 5.6 The ladder hinge is live and self-verified

`C_lad05_s4004`'s training log, which is the loader's own report and not a claim:

```
Loaded 175736 TV pairs from ".../ladder_tv_pairs_safesyn.tsv"
  (weight 0.5, every 50, batch 16, bands=true, band_weights=Some([1.5, 0.5, 0.5, 0.5]))
```

All 175,736 pairs loaded, **zero** out of range (the loud-drop counter added in
this lane prints nothing when the count is zero, and refuses outright when it is
total), bands active with the measured low-q emphasis. The index space and the
`--group` order agree.

### 5.7 The ladder arm at k = 3 — a VARIANCE result on monotonicity, and floors it partly recovers

| arm | k | CID22 | KonJND \|·\| | AIC-3 | composite | contract | `mono` (k=3) | `mono` spread | C6 per seed |
|---|--:|--:|--:|--:|--:|:--:|--:|--:|--:|
| `A_plain` | 3 | 0.8891 ±0.0042 | 0.4997 ±0.0117 | 0.7974 ±0.0059 | 0.8729 ±0.0008 | 4/6 | 0.94602 | 0.00436 | 1642 / 1650 / 1182 |
| `B_nonneg` | 3 | 0.8800 ±0.0049 | 0.4987 ±0.0251 | 0.7878 ±0.0072 | 0.8646 ±0.0020 | **6/6** | 0.93245 | 0.00606 | **0 / 0 / 0** |
| `C_lad05` | 3 | 0.8796 ±0.0074 | 0.4931 ±0.0431 | 0.7850 ±0.0042 | 0.8627 ±0.0054 | **6/6** | **0.94159** | **0.00096** | **0 / 0 / 0** |

**The ladder hinge does what it was aimed at, and it is a variance result as much
as a mean one.** It recovers **67 %** of the monotonicity the architecture cost
(0.93245 → 0.94159 against the control's 0.94602) and **cuts the seed spread
6.3×** (0.00606 → 0.00096). Same shape as the D3 recipe's KonJND finding: the
arm does not add skill so much as remove the downside.

**Floors — all five improve over the architecture alone, none reaches the bar:**

| | rav1e | svt | jpeg | jxl | webp |
|---|--:|--:|--:|--:|--:|
| mentor bar | 0.6410 | 1.0000 | 0.6667 | 0.9615 | 1.0000 |
| `A_plain` | 0.1538 | 0.7778 | 0.5470 | 0.4231 | 0.9658 |
| `B_nonneg` | 0.1880 | 0.6068 | 0.4017 | 0.3205 | 0.8034 |
| `C_lad05` | **0.1966** | **0.6923** | **0.4872** | **0.3974** | **0.8974** |

So the ladder recovers most of the floor the architecture spent (+0.009 / +0.086
/ +0.086 / +0.077 / +0.094) and still lands **below the control on four of five**
and far below every mentor bar. **A7r remains 5 of 5 fail on every arm.**

**Cost of the ladder on top of the architecture: essentially zero on CID22**
(−0.00039, per-seed −0.00207 / +0.00176 / −0.00085) and inside the seed spread on
everything else. It is close to free.

**Two-reference inversions** (`--inversion-truth agree`, pnorm3 @ 0.05, 9,593
reference cells, `unknown = 0` everywhere):

| arm | `mono_agree` (k=3) | encoder-attributed rungs |
|---|--:|--:|
| shipped D | **0.99469** | 15 |
| `peer_ssim2` (mentor) | 0.99160 | 26 |
| `A_plain` | 0.94602 | 15 / 16 / 15 |
| `B_nonneg` | 0.93245 | 13 / 15 / 14 |
| `C_lad05` | 0.94159 | 14 / 15 / 14 |

Every arm clears C1's 0.93 bar; every arm is an order of magnitude worse than the
shipped dial on the axis C1 measures. Attribution is never the excuse — `unknown`
is 0 on all nine cells, so nothing is being charged to "we could not tell".

### 5.8 One near-lossless reading, and the correction to it

`hfnlproxy` **pooled** `srocc_signed` moves **0.3700 → 0.4773 → 0.5022** across
control → architecture → ladder, i.e. **+0.13 over the control** on the axis this
metric is weakest and the product cares about most.

**Two corrections to that reading, both from checking rather than from
inspection.**

**(a) It is not a win over shipped D.** The paired bootstrap on the sampled
per-pair block (n = 5,000, B = 2,000) reads `C_lad05_s4004` **0.48691** against
shipped D's **0.48477**: `+0.00213, CI [−0.01100, +0.01522]` — a **tie**. The k=3
mean of 0.5022 against D's 0.4921 is a mean-vs-point comparison the interval does
not support.

**(b) It is the POOLED statistic, and the per-ref one disagrees in DIRECTION.**
I first wrote "(per-ref)"; it is `srocc_signed`, pooled over all 11,356 rows. The
per-ref mean — which is the axis `CLAUDE.md`'s scoreboard column "HF-NL/ref"
reports — reads:

| | pooled `srocc_signed` | **per-ref mean** |
|---|--:|--:|
| `A_plain_s4004` | 0.33013 | 0.68263 |
| `C_lad05_s4004` | 0.49667 | 0.78798 |
| shipped D | 0.49210 | **0.83062** |

So on the pooled axis the ladder arm ties shipped D; **on the per-ref axis it
loses to it by 0.043**. Both are true of the same predictions, and a
near-lossless claim that does not name its statistic is not a claim. Recorded
because I labelled it wrong first.

`C_lad05_s4004` vs shipped D on the rest: CID22 **+0.01620 WIN**
[+0.01259, +0.01982]; KonJND −0.02590 **tie**; AIC-3 +0.00974 tie; LIVE +0.00349
tie; CSIQ **+0.05704 WIN**. TID (+0.108) and KADID (+0.123) are **train==val for
this recipe** — both are training groups — so those are memorization guards, not
skill, and must not be read as wins.

### 5.9 The hinge weight is a variance/mean trade, and neither end reaches A7r

| arm | `mono` (k=3) | spread | rav1e | svt | jpeg | jxl | webp | CID22 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| mentor bar | — | — | 0.6410 | 1.0000 | 0.6667 | 0.9615 | 1.0000 | — |
| `A_plain` | 0.94602 | 0.00436 | 0.1538 | 0.7778 | 0.5470 | 0.4231 | 0.9658 | 0.8891 |
| `B_nonneg` | 0.93245 | 0.00606 | 0.1880 | 0.6068 | 0.4017 | 0.3205 | 0.8034 | 0.8800 |
| `C_lad05` (w = 0.5) | 0.94159 | **0.00096** | 0.1966 | 0.6923 | **0.4872** | 0.3974 | **0.8974** | 0.8796 |
| `D_lad20` (w = 2.0) | **0.94648** | 0.00861 | **0.2222** | **0.7179** | 0.4615 | 0.3974 | 0.8632 | 0.8787 |

**`w = 2.0` fully recovers the control's monotonicity** (0.94648 vs 0.94602) —
the architecture's C1 cost is entirely repayable by the ladder hinge — **but at
9× the seed spread** (0.00861 vs `w = 0.5`'s 0.00096). The two ends of the sweep
buy different things: 0.5 buys **reproducibility** (and the best jpeg/webp
floors), 2.0 buys **mean monotonicity** (and the best rav1e/svt floors). Neither
is dominated, and **neither moves A7r off 5 of 5**.

Both keep **contract 6/6 on every seed**, and the CID22 spread between them
(0.8796 vs 0.8787) is inside its own seed spread — the hinge weight is not a rank
lever in this range.

### 5.10 The per-ref panel says something the pooled one hides

Within-image (per-reference) means, k=3 — a different statistic of the same
predictions, and the one the board's `HF-NL/ref` column and the CID22-per-ref
selection descriptor use:

| arm | cid22 | aic3 | csiq | live | **hfnlproxy** | **imazen26** | **nonphoto** |
|---|--:|--:|--:|--:|--:|--:|--:|
| `A_plain` | 0.9603 | 0.9551 | 0.9659 | 0.9632 | 0.6897 | 0.9260 | 0.9257 |
| `B_nonneg` | 0.9553 | 0.9464 | 0.9550 | 0.9595 | **0.7777** | 0.9371 | 0.9318 |
| `C_lad05` | 0.9551 | 0.9444 | 0.9655 | 0.9626 | **0.7872** | **0.9472** | **0.9394** |
| `D_lad20` | 0.9552 | 0.9444 | 0.9640 | 0.9637 | **0.7879** | 0.9459 | 0.9389 |
| shipped D | — | — | — | — | **0.8306** | — | — |

Three readings the pooled panel does not give:

1. **The constraint costs about half as much within-image as pooled.** CID22
   per-ref falls 0.9603 → 0.9551 (**−0.0052**) against the pooled −0.0091. The
   constraint hurts between-image scale more than within-image ordering, which is
   the ordering a codec dial actually walks.
2. **`imazen26` and `nonphoto` — the two ssim2 north stars for non-photo content,
   which are first-class gates — IMPROVE**: 0.9260 → 0.9472 and 0.9257 → 0.9394
   with the architecture + ladder. Both the constraint and the hinge help there;
   the pooled panel shows the opposite sign on `nonphoto` (0.9585 → 0.9461).
3. **The near-lossless gain is real and still short of shipped D.** `hfnlproxy`
   per-ref goes 0.6897 → 0.7777 (architecture, +0.088) → 0.7872 (ladder, +0.010),
   against shipped D's **0.8306**. A large move in the right direction on the
   axis this metric is weakest, that does not close the gap to the incumbent.

### 5.11 `E_plainlad` — the attribution control, and it separates the two mechanisms cleanly

This arm is the ladder hinge with **no architecture change**. The plan added it
with the budget freed by dropping rev2, on the grounds that without it a win by
`nonneg + ladder` could not be attributed. It earns its place:

| arm | contract | C5 outside | C6 | `mono` (k=3) | spread | A7r | CID22 | spread |
|---|:--:|--:|--:|--:|--:|--:|--:|--:|
| `A_plain` | 4/6 | 38 | 1491 | 0.94602 | 0.00436 | 5.0 | 0.8891 | 0.0042 |
| `B_nonneg` | **6/6** | **0** | **0** | 0.93245 | 0.00606 | 5.0 | 0.8800 | 0.0049 |
| `C_lad05` | **6/6** | **0** | **0** | 0.94159 | 0.00096 | 5.0 | 0.8796 | 0.0074 |
| `D_lad20` | **6/6** | **0** | **0** | 0.94648 | 0.00861 | 5.0 | 0.8787 | 0.0049 |
| `E_plainlad` | 4/6 | 38 | 513 | **0.95201** | 0.00383 | **4.7** | **0.8855** | **0.0024** |

**The two mechanisms are separable, and they interfere.**

- **C5 and C6 come from the ARCHITECTURE, and only from it.** The ladder alone
  leaves identity 38/38 outside the band and **513** cells above identity. It
  moves C6 (1,491 → 513, a 66 % reduction) but does not close it, and it does not
  touch C5 at all. No amount of ladder supervision substitutes for the
  constraint.
- **Monotonicity and the floors come from the LADDER, and it works BETTER
  without the architecture.** `E_plainlad` posts the wave's best `mono`
  (**0.95201**, above the control's 0.94602 and above every constrained arm) and
  its best jpeg floor (0.5812 vs the control's 0.5470).
- **The architecture COSTS what the ladder buys.** That is why
  `nonneg + ladder` lands at 0.94159/0.94648 rather than at 0.95201 — the hinge
  spends most of its effect repaying the constraint instead of improving on the
  control.
- **The ladder alone is the cheapest arm on rank** — CID22 0.8855 (−0.0036 vs
  control) with the **tightest seed spread in the wave** (0.0024 against the
  control's 0.0042).

**The one A7r movement in the entire wave is not a result.** `E_plainlad` reads
A7r 4.7 because **webp hit exactly 1.0000 against its 1.0000 bar on one seed of
three** (`s4005`); the other two seeds read 0.9744 and 0.9231. A single-seed
boundary touch on the easiest codec is a boundary touch, not a mechanism. The
best any arm gets on the hard codecs is `rav1e 0.2564` against a 0.6410 bar and
`jxl 0.4231` against 0.9615.

**Conclusion on A7r, now with an attribution behind it: none of architecture,
ladder supervision, hinge weight, or their combination moves it.** That is
consistent with the fastclass2 campaign's own finding that 35 recipe-axis cells
cleared it zero times, and it extends that null to two mechanisms the campaign
did not test.

### 5.12 `F_nonneg32` — the H32 variant is the best constrained arm on rank

| arm | CID22 | spread | composite | contract | M3a (k=3) |
|---|--:|--:|--:|:--:|--:|
| `A_plain` | 0.8891 | 0.0042 | 0.8729 | 4/6 | 0.7823 |
| `B_nonneg` (H128) | 0.8800 | 0.0049 | 0.8646 | **6/6** | 0.8016 |
| `F_nonneg32` (H32) | **0.8824** | **0.0036** | 0.8655 | **6/6** | **0.8212** |

Halving the hidden width **costs nothing and buys the best constrained CID22,
the tightest spread, and the best M3a in the wave** — consistent with the
campaign's measured "capacity is not a lever" and with its H32 byte savings
(30–47 % smaller). If a constrained model ever ships, it should be the H32 one.

### 5.13 M3a — every constrained arm IMPROVES diffmap coherence

| arm | M3a (k=3) |
|---|--:|
| `A_plain` | 0.7823 |
| `E_plainlad` (ladder only) | 0.7996 |
| `B_nonneg` | 0.8016 |
| `D_lad20` | 0.8113 |
| `C_lad05` | 0.8191 |
| `F_nonneg32` | **0.8212** |

The constraint and the hinge both help G-STEER, and they compose: the control is
last and the constrained+H32 arm is first, **+0.039** over it. All are `silver`
(≥ 0.78); none reaches `gold` (≥ 0.85). Nothing here was aimed at M3a, so this is
a free by-product rather than a claim.

### 5.14 ⛔ THE REGISTERED SELECTION RULE PICKS THE CONTROL — because it cannot see the contract

`freeze_check --select --seed-group --min-k 2 --floor-basis all` over all 18
cells:

```
SELECTED: 8ad90f29c3a8 — a RECIPE, k=3, 8 floors passed by every seed
  mean selection_composite 0.9850 (spread 0.9833–0.9862)
  members: A_plain_s4004, A_plain_s4005, A_plain_s4006
```

**The control wins, and the reason is precise: the one floor every constrained
arm misses is `cid22`.** `B_nonneg_s4004` reads 7/8 with `split floors: cid22` —
the −0.0091 constraint cost is just enough to drop under the profile's CID22
floor. Every other floor passes, *including* `dial`, where it is strictly better
than the control (mono 0.97128 vs 0.96457, `tied` 0.00000 vs 0.00718) and on
`hfnl` (0.78433 vs 0.68263).

**The rule's floor set is `bandtail, breadth, cid22, dial, dialrange, hfnl,
konjnd, nonphoto`. C5, C6 and A7r are not in it.** So a model that takes the dial
contract from 4/6 to **6/6 on every seed** is scored identically to one that does
not, and loses on a CID22 hair. That is not a defect in this lane's arms — it is
a **gap in the selection rule**, and it is worth registering: the rule was written
before the G-ADDR contract existed as a gate, and it never absorbed it.

### 5.15 First anchor-ladder cell — the floors DO respond to floor-reaching data

`G_anchorlad_s4004` (plain net + the anchor group's floor-reaching ladders,
`--tv-weight 2.0`, band 3 = the floor window at 31.4 % of the ladder gradient):

| | `A_plain_s4004` | `E_plainlad_s4004` (ladder, no anchor) | **`G_anchorlad_s4004`** |
|---|--:|--:|--:|
| `mono` | 0.94868 | 0.95431 | **0.96100** |
| C6 (cells above identity) | 1,642 | 1,539 | **102** |
| avif-rav1e | 0.1538 | 0.2564 | 0.1538 |
| avif-svt | 0.8205 | 0.7436 | **0.9231** |
| jpeg | 0.5641 | 0.6154 | 0.5897 |
| jxl | 0.3846 | 0.4231 | **0.4615** |
| webp | 0.9744 | 0.9744 | 0.9487 |
| CID22 | 0.89036 | — | 0.88474 |

**The hypothesis has real support.** Floor-reaching training ladders move the
things §5.11 said were immovable:

- **`mono` reaches 0.96100** — the best in the entire wave, above every ladder
  arm and every constrained arm.
- **C6 falls 1,642 → 102** on a net with NO architectural constraint. A 94 %
  reduction from data alone, where the loss alone managed 6 %.
- **avif-svt 0.8205 → 0.9231** against a 1.0000 bar, and **jxl 0.3846 → 0.4615**
  against 0.9615.
- CID22 costs only **0.0056**.

**And the pre-registered limit holds exactly as written.** `avif-rav1e` is
**0.1538 — bit-identical to the control**, because the anchor set contains no
rav1e ladders. §9 said this arm structurally cannot move rav1e and that a rav1e
failure would not be evidence against the hypothesis; it did not move, and it is
not.

**A7r is still 5 of 5**, because "moved a lot" is not "reached the bar": svt is
0.077 short of 1.0, jpeg 0.077 short of 0.6667, jxl 0.500 short of 0.9615.

*(RESULTS — the anchor arms at k=3, and `H_anchorlad` vs `D_lad20`, pending.)*

---

## 6. RESULTS

*(pending)*
