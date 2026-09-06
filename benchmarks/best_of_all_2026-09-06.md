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

**Negative control (the important half).** At `main@origin` `0c6307a7`, running
the same test file: **4 of 5 FAIL**. The 5th — "rank-only recipes are
DISTANCE-shaped on every path" — passes there and here, which is the point.

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

*(RESULTS — filled when the wave lands.)*

---

## 6. RESULTS

*(pending)*
