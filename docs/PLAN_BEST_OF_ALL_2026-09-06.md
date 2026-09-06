# PLAN — BEST-OF-ALL: a CONSTRAINED MLP on the 228 servable slots (2026-09-06)

**Status: PRE-REGISTERED.** Written and pushed BEFORE any code change, per the
project's pre-registration discipline. Everything below that is not marked
MEASURED is a hypothesis or a decision, not a result. Results land in
`benchmarks/best_of_all_2026-09-06.md` and in the RESULTS section at the end of
this file.

Lane: `zensim--bestofall` (sibling jj workspace). Artifacts:
`/mnt/v/output/zensim/best-of-all-2026-09-06/`.

---

## 0. The thesis, in one paragraph

`benchmarks/fastclass2_campaign_2026-09-05.md` measured a 228-input MLP that
**ranks with the 944 leaders and is faster than shipped D** — and that **fails
the dial contract on identity (90.9 vs the required 100), puts 1,642 of 9,593
grid cells above identity, and fails A7r on 5 of 5 codecs.** Those are *weights*
properties. `benchmarks/dial_addressability_gate_2026-09-04.md` §10.3 proved the
matching either/or on the shipped-B lineage: **no monotone output spline can
satisfy C2 and C6 at once** when the raw ordering puts real cells above a perfect
copy. So the dial cannot be repaired downstream of the weights. This plan makes
the dial properties **structural**: an architecture in which
`g(0⃗) = 0` and `g ≥ 0` hold *bit-exactly, for every input, by construction*, so
identity is a pin and "nothing above identity" is a theorem rather than a
measurement — and a **ladder-ordering loss** that supervises the within-ladder
direction the dial actually needs, on pairs a *second reference metric* agrees
are materially ordered.

The exam is unchanged: it must rank with the leaders, pass the dial contract and
the per-codec floors and the two-reference inversions, be fast, bounded,
servable, and ship with the tree corruption head.

---

## 1. What I read first (and the four owner defects it surfaced)

Read in full before writing this: `benchmarks/fastclass2_campaign_2026-09-05.md`,
`fastclass_distill_wave_2026-09-04.md`, `replication_wave_2026-09-05.md`,
`subset_quality_study_2026-09-04.md`, `d_id100_2026-09-04.md`,
`dial_addressability_gate_2026-09-04.md` §14–17, `inversion_truth_2026-09-05.md`,
`ladder_instrument_2026-09-05.md`, `rev2_refit_2026-09-06.md`,
`rev2_d_arms_2026-09-06.md`, `corruption_head_serving_2026-09-06.md`,
`docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md`, `docs/PLAN_CRUFT_PURGE_2026-09-06.md`,
`docs/WAVE_PLAYBOOK.md`, `benchmarks/d_ship_flip_2026-09-05.md` §6, and the
trainer itself (`zensim-validate/src/bin/zensim_mlp_train.rs`,
`zensim-validate/src/mlp_train/`, `zensim-train-core/src/`).

Reading the trainer produced four defects, all **read from source**, all in the
same family — *a convention with no owner*:

**D1 — OUTPUT POLARITY IS APPLIED AT EXACTLY ONE OF EIGHT POLARITY-SENSITIVE
SITES.** `mlp_train/mod.rs:2088` derives `rank_target_sign` (`-1.0` when any
group carries an absolute/MSE term, else `+1.0`) with a long, correct comment
explaining that the RankNet term is DISTANCE-shaped (higher quality → LOWER `y`)
while the absolute term is SCORE-shaped, and that mixing them naively makes a
corpus rank backwards — MEASURED 2026-07-15 as "+0.6393 → −0.3454". It is then
used at **one** site (`:2556`). Every other RankNet site computes the raw
`(mos_a - mos_b).signum()`:

| site | line | reached by |
|---|---|---|
| `train_mlp_strategy` main loop | 2556 | plain path, `--minibatch-size 1` — **has the fix** |
| `train_mlp_pool_head_with_tv` | 3399 | `--pool-head` |
| `train_mlp_hybrid_head_with_tv` | 4286 | `--hybrid-head` |
| `run_parallel_minibatch` | 5766 | **plain path** whenever `--minibatch-size > 1` |
| `run_minibatch_with_nin` | 5962 | **plain path** whenever `--norm-in-norm-weight > 0` |
| `train_mlp_per_sample_alpha_head` | 8095 | `--per-sample-alpha-head` (the only depth-2 / skip path) |

The TV hinge carries the same assumption independently and in prose
(`:2694` "Rust trainer output is distance-like (lower = better)"), as does the
triplet term (`strategy.rs:83` "model QUALITY scores"), the α-head MSE term
(`:8206`, regresses `y` onto `mos` directly = score-shaped), and the anchor /
pjnd / konjnd-aggregation terms. **Hypothesis (to be MEASURED in A1): this is the
mechanism behind the campaign's α-head ordering inversion** — an α-head recipe
that carries `--mse-weight` mixes a score-shaped absolute term with a
distance-shaped rank term on a path where nothing reconciles them, which is
exactly the failure `rank_target_sign` was written to prevent on the one path it
guards.

**D2 — `run_parallel_minibatch` SILENTLY DROPS THE ABSOLUTE TERM ENTIRELY.** It
is pure RankNet + PWRC (`:5694`–`:5900`): no MSE, no monotonicity hinge, no
polarity reconciliation. So on the plain path, `--minibatch-size K>1` throws away
both the MSE term and the polarity fix without a word. (`parallel = parallel_batch
&& k > 1`, `:2372`; the default `k = 1` keeps today's board bakes on the guarded
sequential branch — so this is a latent trap, not a claim that any shipped bake
is wrong. Which bakes set `K>1` **with** an absolute term is a MEASUREMENT this
plan owes, in A1.)

**D3 — `--n-hidden-layers` AND `--skip-connection` ARE SILENT NO-OPS OFF THE
α-HEAD PATH.** `use_2layer` / `use_skip` are read at exactly two lines
(`:6611`, `:6612`), both inside `train_mlp_per_sample_alpha_head`. The dispatcher
already fails loud for `--ema-decay`, `--hard-pair-frac`, `--dro-eta`,
`--listwise-weight`, `--monotonicity-reg`, `--mse-weight` and the triplet flags
(`:1924`–`:1999`) for precisely this reason; these two were missed. A run that
asks for depth 2 without `--per-sample-alpha-head` gets a **1-layer net** and a
bake byte-identical to one that never asked.

**D4 — `--leaky-alpha` IS A TRAIN/SERVE DIVERGENCE.** The trainer takes
`--leaky-alpha` as a hyperparameter, but every bake emitter hardcodes
`Activation::LeakyRelu` (`zensim-train-core/src/mlp.rs:47`,
`mlp_train/mod.rs:5371`, `pool_head.rs:602`, `hybrid_head.rs:633`), and the
runtime applies a **hardcoded** `LEAKY_RELU_ALPHA` for that byte
(`zenpredict/src/inference.rs:305–311`). So any `--leaky-alpha ≠ 0.01` trains one
function and serves another, with no warning. `Activation::Relu` (byte 1) exists
in the wire format and in the runtime and is currently unreachable from this
trainer.

D4 is load-bearing for this plan: the architecture below **needs ReLU**, and
needs the served activation to be the trained one.

---

## 2. The architecture — `--nonneg-distance`

### 2.0 The polarity fact this architecture must respect (MEASURED, from the recipe)

The winning 228 recipe declares `:both` (rank **and** absolute) on 4 of its 6
groups — `safesyn`, `cid22_train`, `bigcodec`, `konjnd`. So on the plain path
`rank_target_sign = -1.0` and **the model is SCORE-shaped**: raw output rises
with quality, and the absolute term regresses raw output onto `human_score ×
100`. The α head, which never applies `rank_target_sign`, stays on the legacy
DISTANCE convention — which is exactly the mechanism predicted in §1/D1 and
exactly what the campaign measured: **the 2-layer α head is the campaign's best
CID22 *ordering* (|−0.8921| beats the plain path's +0.8863) with the sign
backwards**, and `bake_dial_refit pack` then fails because the output spline is
monotone increasing by construction and cannot express a decreasing map.

The architecture below is therefore written in the **score-shaped** convention
the winning recipe actually trains in.

### 2.1 The choice, pre-registered

The brief offers softplus / ReLU² / squared-norm. **I pre-register none of them.
I pre-register the one form that is exactly expressible in the SHIPPED wire
format and the SHIPPED runtime with zero changes to either:**

> **N1 — `score(x) = pin − g(x)`, realized as a ReLU encoder with zeroed hidden
> biases and a sign-constrained final layer whose bias IS the pin.**
>
> - `scaler_mean := 0⃗` (scale-only standardization; `scaler_scale` unchanged)
>   ⇒ the identity input `x = 0⃗` standardizes to `0⃗` exactly
> - hidden biases frozen at exactly `0.0` (`b1`, and `b2_enc` if present)
>   ⇒ `h(0⃗) = ReLU(0⃗) = 0⃗` exactly
> - hidden activation **ReLU** (`leaky_alpha := 0`, baked as `Activation::Relu`)
>   ⇒ `h ≥ 0` for every input
> - final-layer weights projected **`≤ 0`** after every Adam step
> - final bias frozen at **`pin`** (`--nonneg-pin`, default `100.0`)
>
> ⇒ `g(x) := pin − raw(x) = −(w_out · h) ≥ 0` for every `x`, and
> `raw(0⃗) = pin` **bit-exactly** (`0·w = 0` exactly in f32, f16 and i8;
> `ReLU(0) = 0`; `dot(0⃗, w) = 0`; the frozen bias passes through untouched).
>
> **`raw(0⃗)` is the argmax of `raw` over the entire input space, by
> construction.**

**Why the pin lives in the output bias.** The brief's `score = pin − g(x)` needs
`pin` *somewhere*. Putting it outside the net would need a new wire field and a
change in every serving consumer. Folding it into the **frozen** output bias
costs nothing: `raw(0⃗) = pin` is still a bit-exact constant, the guarantee is
unchanged, the absolute (MSE) term — which regresses raw output onto a target in
`[0, 100]` — becomes *satisfiable* (it is not, with a strictly non-positive raw
output), and the runtime is untouched. The claim "no bias in any layer" is
therefore stated precisely as **hidden biases frozen at 0, output bias frozen at
the pin**; that is what makes `g(0⃗) = 0` and `g ≥ 0` hold, and nothing weaker
would.

**Why not softplus / ReLU² / squared-norm.** Each needs a new `Activation` or a
new head in `zenpredict` — a wire-format change plus a change in every serving
consumer — for a property N1 already gets exactly. `softplus(0) = ln 2` also
needs a constant offset baked somewhere, which is one more place for the pin to
drift.

### 2.2 What this buys, and why it is a theorem rather than a measurement

MEASURED baseline for the winner `S372_S228_H128_p` (seed 4004, `_id100` cell,
`gaddr_S372_S228_H128_p_s4004_id100_ladder.json`):

| gate row | the fastclass2 228 MLP | under N1 |
|---|---|---|
| **C5** identity ∈ [97.5, 100] | **90.9368** — 38 of 38 rows outside | `raw(0⃗) = pin` bit-exactly ⇒ `spline(pin) = 100` exactly, by construction |
| **C6** nothing above identity | **1,642 of 9,593 cells above (17.1 %)** | `raw ≤ raw(0⃗)` ∀x and spline monotone increasing ⇒ `score ≤ 100` ∀x, by construction |
| **C2 ⊻ C6** either/or (gate doc §10.3) | structurally unsatisfiable; this bake is **worse than shipped B** (17.1 % vs 6.01 %) | **does not arise** — no cell can out-score identity, so the compression that forced the tie is gone |
| **A7r** floor representability | **5 of 5 codecs fail**; `n_fail_clamp = 0` on every codec, `n_fail_order = 32` on avif-rav1e | **NOT claimed.** A7r is an ORDERING failure at the bottom of ladders. §3's ladder loss is the mechanism aimed at it; the architecture alone is not. |

Note the honest split: C5/C6 fall out of the architecture; **A7r and the C3/C4
floor rows do not.** A7r is what the ladder loss is for; C3/C4 depend on the
spline's low end being fit on negative-rich data (the `negrich` half of the
`d_id100` chain, applied unchanged). Three rows from one mechanism, not six.

The campaign's own reading of the same fact — *"if your architecture constraints
can make `raw(0⃗)` the argmax, you close C5, C6 and probably A7r's ordering
failures at once"* — is the hypothesis this lane tests. C5 and C6 are proved
here; the A7r half is a **prediction that the wave will confirm or refute.**

Also on the record, so a C5 failure is not over-read: `Zensim::compute`
short-circuits byte-identical input to exactly 100 before the model runs
(`metric.rs` `.mark_identical()`). C5 governs the **near**-identity regime —
one-bit differences, lossless re-encodes, q=100 JPEG — not `zensim(x, x)`.

### 2.3 The honest cost, pre-registered as a finding either way

At one hidden layer, `g` = a non-negative combination of `ReLU(linear)` is a
**convex** function of the standardized features. That is a real restriction, and
the plain path is 1-hidden-layer only (D3) — and `--keep-features` is refused
outright with `--n-hidden-layers ≥ 2`, so a 2-layer 228 student is today only
buildable through the α head, i.e. the inverted one. (A1 is what would unblock
that; building it is **not** in this lane's arms.)

The arm table crosses the constraint against the control at **matched seeds,
matched capacity, matched features**, and the reported constraint cost is
`rank(nonneg) − rank(plain)` at those settings. **Whichever sign that number has,
it is the finding.** A constrained model that ranks with the leaders is the
deliverable; one that costs 0.02 CID22 is a priced trade for the user; one that
costs 0.10 is a measured refutation of §0 and I will say so in these words.

`--skip-connection` is **incompatible** with `--nonneg-distance` — the skip term
`x · w_skip` is sign-free because scale-only standardized `x` is sign-free, which
breaks `raw ≤ pin`. The trainer fails loud on the combination rather than emit a
bake whose guarantee silently does not hold. (No loss: the campaign MEASURED
`--skip-connection` byte-for-byte inert on rank — every axis identical to 4 dp.)

---

## 3. The ladder-ordering loss — extend the OWNER, do not add a flag

The brief asks for `--ladder-hinge <w>`. **`TvRegularizer` already is that
loss** — `mlp_train/mod.rs:1727`–`:1764`: a within-ladder pairwise hinge over
adjacent severity levels, with an anti-collapse `margin`, per-pair band weights,
`apply_every` and `batch`, wired on the plain path (`:2691`), the pool head
(`:3597`), the hybrid head (`:4532`) and the α head (`:8666`). Its own comment
reasons about ladder collapse and dynamic range. The distill wave independently
names it as **the untried W3 lever**, with the shape hint that the hinge should
be weighted toward LOW quality: `q<50` carries 28 of the 57 material inversions
at **0.03526** per pair against `q≥85`'s 0.00727 and `q 50–85`'s 0.00793 — a 3×
higher rate.

Adding a second flag for the same loss would be a **duplicate implementation**,
which this repo bans outright (CLAUDE.md, "NO DUPLICATE IMPLEMENTATIONS — one
owner per task, extend it or don't do it"). That rule outranks a flag name in a
brief. So:

- **the loss is `--tv-weight` / `--tv-margin` / `--tv-apply-every` / `--tv-batch`
  / `--tv-band-weights`** (read `--ladder-hinge w` as `--tv-weight w`);
- **the new artifacts are (a) the BUILDER and (b) two owner extensions the TV
  path needs before it can carry this job.**

### 3.1 Two owner extensions the TV path needs (A5)

**`--tv-margin` is α-head-only.** Read at `:8666` and nowhere else; the plain
path's hinge (`:2735`) is `max(0, y_hi − y_lo)` with no margin. A pure hinge is
minimized by collapsing every ladder flat — the `TvRegularizer` doc says exactly
this — and flat ladders are `tied`, which is **C2**. Wiring `--tv-margin` on the
plain path is a prerequisite, not a nicety. Default `0.0` ⇒ byte-identical.

**The TV hinge is polarity-blind, and this recipe is score-shaped.** `:2694`
states the assumption in prose — *"Rust trainer output is distance-like (lower =
better)"* — and pushes `y_hi < y_lo`. Under the winning recipe's
`rank_target_sign = −1` (§2.0) higher quality must get a **higher** `y`, so the
TV hinge as written would train the ladder **backwards**. **A1 is therefore a
hard prerequisite for the ladder arm**, not an independent cleanup: adding a
ladder loss to this recipe without the polarity owner would produce a
confidently-wrong model.

### 3.2 The builder, the data, and the materiality rule

The TV pairs file is `lo_idx <TAB> hi_idx [<TAB> band_id]`, header starting
`lo_`, indices into the **concatenated group feature rows in `--group` order**
(loader: `zensim_mlp_train.rs:3137`–`:3210`). `safesyn` is group 0 in the
winning recipe, so its global indices are its local row indices.

**The training-side ladder exists and is a POSITIONAL join (MEASURED).**
`/mnt/v/zen/zensim-training/2026-05-16/safesyn_with_iwssim.csv` is row-aligned
with `canonical-2026-05-21/train/safesyn.parquet`: both **196,086** rows, **0**
`ref_basename` mismatches across all of them, and every `human_score` equals
`cpu_ssimulacra2/100` (100,932 rows) or `gpu_ssimulacra2/100` (95,154 rows) to
<1e-9 with **0** unmatched. It carries `codec`, `quality`, `size_bytes`,
`cpu_ssimulacra2`, `gpu_ssimulacra2`, `cpu_butteraugli`, `gpu_butteraugli`.

Structure: **19,259 ladders** over 6 codecs, a 16-step q grid
(`5,10,15,20,25,30,40,50,60,70,75,80,87,90,95,100`), **19,170 with ≥2 steps** ⇒
roughly **176k** adjacent pairs before filtering.

Pair construction, pre-registered:

1. Group rows by `(ref_basename, codec)`; sort by `quality`.
2. **Saturation dedup first.** The training q grid starts at 5 and steps by 5, so
   zenjpeg's known q0–q10 plateau is only partly present and *nothing has
   verified that q5 and q10 are distinct settings*. Collapse consecutive steps
   with identical `size_bytes` before forming pairs. (The ladder instrument keys
   on `encode_sha`; that column does not exist here, so `size_bytes` is the
   available proxy and is declared as such.)
3. Form **adjacent** pairs only.
4. **Materiality gate — the reference must order them by ≥ 0.5 ssim2 points.**
   This is the same `ENCODER_SSIM2_MARGIN_PT = 0.5` / `MATERIAL_INV_PT` the gate
   grades the dial with, so the loss is supervised on exactly the moves the exam
   calls material and on no others.
5. **Arm 1 is ssim2-margin-only and is declared as such.** The sidecar also
   carries butteraugli, so a second arm can require the two references to AGREE
   on direction (the `encoder_inversion` shape). Its variant is **not** recorded
   in the CSV header, so before it can be used the builder must identify it
   empirically the way the gate's own lane did; until then the agreement arm is
   **NOT MEASURED**, not silently skipped.
6. **Low-q emphasis** via `band_id` + `--tv-band-weights`, following the measured
   3× inversion-rate concentration at `q<50`.
7. `_MANIFEST.json` beside the TSV: source paths + sha256 + row counts + group
   order + kept/rejected pair counts + the exact margin rule.

Coverage is a MEASUREMENT the builder reports, never an assumption.

### 3.3 One silent-drop the builder forces closed

The TV loader **silently drops out-of-range pairs** (`continue`, `:3177`). A
builder that computed indices against a different group order would train with a
silently-decimated ladder term and produce a plausible bake. The loader will
**count the drops, print the count, and panic if every pair was dropped.** Files
with zero out-of-range pairs are unaffected, so no existing recipe moves.

### 3.4 `--identity-rows`

Implemented as the brief asks, **default off**, as a redundant guard: `N`
synthetic zero-feature rows whose target is the pin, entering the absolute term.
Under N1 its loss contribution is **exactly `0.0`** (because `raw(0⃗) = pin`
bit-exactly), so its gradient is exactly zero and the bake is byte-identical.
That is the test — `identity_rows_are_a_no_op_under_nonneg_distance` — and it
proves the architecture's central claim rather than decorating it.

Note this is a *training-side* guard and is distinct from the **pack-side**
identity anchor rows (§6), which are what actually pin the spline knot.

---

## 4. PHASE A — owner work (failing-first tests; default paths byte-identical)

Every item below lands with (a) a test that FAILS on the parent commit and passes
after, and (b) a byte-identity control proving the default path did not move.

**A1 — give output polarity ONE owner.** A single derivation
(`OutputPolarity::{Distance, Score}`, computed once from the group loss modes)
threaded to every polarity-sensitive site in §1/D1: the six RankNet sites, the
two TV hinges, the monotonicity hinge, and the α-head absolute/anchor/pjnd/
konjnd-aggregation/triplet terms. Default `Distance` ⇒ **every rank-only recipe,
which is every board bake, is byte-identical**. Tests:
- `output_polarity_is_consistent_across_all_training_paths` — a synthetic corpus
  trained through the plain path, the plain path at `--minibatch-size 32`, the
  pool head, the hybrid head, and the α head at depth 1 **and** depth 2, with and
  without an absolute term; every arm's raw output must carry the **declared**
  polarity. Pre-fix this fails on the α head and on both minibatch helpers.
- `alpha_head_raw_output_polarity_matches_plain_path_at_depth_1_and_2` — the
  brief's requested proof, stated in the codebase's own convention.
- byte-identity control: a rank-only recipe reproduces its parent-commit bake
  sha256 exactly.
- The α head's ordering inversion is then **re-measured** at depth 1 and depth 2
  on the campaign's own recipe. If the polarity fix does not move it, the
  hypothesis in D1 is FALSIFIED, I say so in this file, and I bisect further —
  I do not ship a claimed fix that did not fix anything.

**A2 — `--nonneg-distance`** per §2.1, on the plain path *and* the α-head path
(the projection, the bias freeze and the zeroed scaler mean are path-agnostic;
only "which weights are the final layer" differs). Fails loud on
`--skip-connection`. Tests:
- `nonneg_distance_output_is_exactly_zero_on_the_zero_vector` — `to_bits()`
  equality against `+0.0`, in the trainer AND after a round-trip through
  `zenpredict::Predictor` from the baked bytes, at f32 / f16 / i8.
- `nonneg_distance_output_is_nonnegative_on_random_inputs` — 100k random vectors
  including extreme and adversarial ones; `y ≥ 0` with no exceptions.
- `nonneg_distance_holds_after_pack_and_prune` — the guarantee must survive
  `bake_dial_refit pack` (zerobias + dead-column pruning + dtype + spline refit),
  which is the form that actually ships.
- `nonneg_distance_default_off_is_byte_identical`.

**A3 — `Activation::Relu` reachable + the train/serve divergence closed (D4).**
`bake_two_layer_znpr_v3` (both copies) gains an explicit hidden-activation
argument; existing callers pass `LeakyRelu` and stay byte-identical. The trainer
maps `leaky_alpha == 0.0 → Relu`, `== 0.01 → LeakyRelu`, and **anything else
fails loud** — the wire cannot express it, so emitting a bake would be a silent
mis-serve. Test: `leaky_alpha_zero_bakes_relu_and_round_trips_bit_identically`
(trainer forward vs `Predictor` forward, `to_bits()` equal) plus
`unrepresentable_leaky_alpha_fails_loud`.

**A4 — `--n-hidden-layers` / `--skip-connection` fail loud off the α-head path
(D3).** Same treatment the dispatcher already gives seven other flags. Test:
`depth_and_skip_are_refused_on_paths_that_ignore_them`.

**A5 — TV loader loud-drop (§3.2)** + the ladder builder + `--identity-rows`
(§3.3) + `zentrain.feature_ids` stamping and `bake_dial_refit densify` on every
bake produced by this lane, with `feature_set_id` computed at the 372 layout via
its one owner (`zensim::feature_set_id::slots_hash8`).

---

## 5. PHASE B — the wave

**Control = the campaign's winner, unchanged**: `S372_S228_H128_p` — the plain
path, `--keep-features scripts/sota944/slice_basic156_peaks.txt` (f0..f227,
contiguous, 156 basic + 72 peaks), `--max-features 372`, six `--group` legs from
`canonical-2026-05-21/train` (**the `_norm` variants — the un-normalised
`cid22_train` and `konjnd-dense` carry targets ~100× the others after
`--target-scale 100`, and nothing crashes**), 40 `--feature-transform` entries,
`--epochs 120 --pairs-per-epoch 50000 --coarse-decay 1e-5`, `--hidden 128`,
seeds **4004 / 4005 / 4006**. Its exact argv is recoverable from
`S372_S228_H128_p_s4004.fulleval.json` → `repro.argv`, and that is the copy this
lane uses — not a retyped one.

- **Sets**: S228 at the v1-372 layout.
- **Features**: `rev1` (the above) × `rev2` (the rev2 S228 tables).
- **Arms**: `plain` (control) × `nonneg` (N1 only) × `nonneg+ladder` (N1 + the TV
  ladder hinge, §3).
- **Capacity**: H32 × H128. **This is a confound control, not a search** — the
  campaign MEASURED capacity is not a lever here (composite moves −0.0038…+0.0021
  across six set×width pairs, inside every per-seed spread) and H256 was
  cancelled on that evidence. Carrying both only ensures the constraint cost is
  not read off a capacity difference.
- **Seeds**: k = 3 at 4004/4005/4006, split via `--init-seed` / `--sample-seed`
  (omitting both is byte-identical to `--seed`). **Pre-flight: verify the trainer
  binary postdates `34b4899f`** — before that fix `zentrain.repro` emitted only
  `"seed"`, which defaults to 1, so split-seed arms silently record `k = 1` and
  `--seed-group` collapses the whole wave into one group.
- 3 arms × 2 capacities × 2 feature revs × 3 seeds = **36 fits**.
- **Recipe base**: the D3 variance-stabilizing recipe. ⚠ **D3's live half does
  not exist at 372.** D3 = `WR4_KON_WITHINREF` + `WR4_HF_WITHINREF`, and the 372
  layout has no `tbig_hf` leg at all, so only the konjnd half is even reachable —
  and `train_372_student.sh` does not implement that env var either. D3 is
  therefore carried as *the recipe lineage*, and the fact that its mechanism is
  largely inert at this layout is stated, not papered over. (D3's own effect was
  a **variance** result — KonJND SD 0.0688 → 0.0079, F = 75.9 — with a mean shift
  of t = 1.73 against a critical value of 4.30. It must not be reported as a mean.)
- **Packing**: `bake_dial_refit pack --neg-tail` with the negrich anchor target
  (`ssim2_gpu`, unclamped) plus **pack-side identity anchor rows** so the spline
  gets a knot at `(pin, 100)` — the `n_id = 21` sizing from the id100 lane
  (21/2,021 ≈ 1.03 % owns `fit_spline_knots`'s `≥ p99` top bin; `n = 38` spills
  and displaces the top real knot). Pruning on by default.
- **Execution**: serialized locally under `~/work/zen/scripts/run-heavy --jobs 8
  --mem 16G`. No paid cloud. Harvested inline (verdict + fulleval), with M3a run
  **after** all fits, exclusively.
- **Progress**: a heartbeat file under the artifacts dir; the terminal condition
  is a single `.done` marker so nothing idles on a poll.

---

## 5b. DEVIATIONS FROM THIS PRE-REGISTRATION (recorded here, not buried)

A pre-registration is only worth something if the deviations are written into
it. Three, all decided before any wave cell was scored:

**D-1 — the `rev2` axis is DROPPED.** Pre-registered as an arm axis; removed on
five measured grounds from `benchmarks/rev2_refit_2026-09-06.md` (all five are
in `benchmarks/best_of_all_2026-09-06.md` §4). The short form: on the fast class
the revision is INERT (every Δ inside its own seed spread), its seed-matched
paired bootstrap SIGN-FLIPS with all three CIs excluding zero, only 3 of 6
groups are substitutable so the arm is 42.6 % in-era, its transform screen is
era-bound, and the two candidate rev2 safesyn tables differ on 51.2 % of cells
over an **unadjudicated AVIF decoder era** under an AVIF backend HOLD. Running
it would have spent half the wave measuring a known-inert, structurally
confounded axis on an input era nobody has ruled on.

**D-2 — `E_plainlad` is ADDED** with the freed budget: the control plus the
ladder hinge and **no** architecture change. Without it, a win by
`nonneg + ladder` cannot be attributed to either half. This is a strengthening
of the design, and it is exactly what the rev2 budget bought.

**D-3 — the capacity axis is asymmetric.** Pre-registered as `{H32, H128}`
across all arms. Run as H128 for every arm plus H32 for the constrained arm only
(`F_nonneg32`). Capacity is MEASURED not to be a lever on this class (composite
moves −0.0038…+0.0021 across six set×width pairs, inside every per-seed spread),
so a full H32 replication would double the wave to re-measure a known null. The
constraint cost is read at MATCHED capacity (H128 vs H128), which is what the
confound control was for; `F_nonneg32` is a bytes/speed variant, not a control.

**D-4 — the pre-registered `identity_rows_are_a_no_op_under_nonneg_distance`
gate was NOT written, and cannot be written as specified.** §3.4 called it "the
test" and "a genuine proof of the architecture's central claim". Under this
wave's transform screen the premise is FALSE — `raw(identity) = 99.6138`, not the
pin, so identity rows carry a real residual and are not a no-op. Writing a test
that asserted otherwise would have encoded the overclaim. The flag's two actual
defects (a `target_scale` division that made the target 1.0 against a raw pin of
100, and an `Mse` group that silently flips the run's polarity) were found by
review instead, and both are fixed. See §2.4 of the record.

**D-5 — A1's scope was cut.** §4/A1 pre-registered threading the owner to "the
six RankNet sites, the two TV hinges, the monotonicity hinge, **and the α-head
absolute/anchor/pjnd/konjnd-aggregation/triplet terms**." The last clause was not
done: the anchor MSE, PJND passthrough, konjnd-aggregation, cross-codec
rank-preserve, ListMLE and ordered-probit triplet terms still carry their own
hard-coded assumptions. `for_groups` does not derive `Score` from any of their
weights, so a recipe with `--anchor-loss-weight` and no MSE still mixes a
score-shaped absolute term with a distance-shaped rank term. **Latent, not live**
— all 10 stored α-head recipes carrying those terms also carry `mse_weight > 0`,
which derives `Score` — but it is a scope cut and it belongs here rather than in
a footnote.

Not deviations, but worth stating because the plan implied otherwise:

- **`--ladder-hinge` is `--tv-weight`.** The plan already said this; the wave
  confirms the owner needed two extensions first (`--tv-margin` on the plain
  path, and polarity-awareness), and those landed.
- **`zentrain.feature_ids` is stamped by `densify`, not the trainer.** The plan
  listed trainer-side stamping as an A-item; it is not the trainer's job and
  adding it would have been a duplicate. `densify` runs in the wave as a
  servability artifact.
- **The SCORED bake is the PACKED one, not the densified one** — MEASURED, see
  `benchmarks/best_of_all_2026-09-06.md` §5.1(b): a densified 228-caller bake
  reads C3/C4/C5/C6 as NOT MEASURED because every registered probe is 372-wide.

---

## 6. PHASE C — gates (all pre-registered; none may be relaxed)

**The bar moved after several of the records the brief cites were written, and
this plan grades against the CURRENT one.** Verified in source and registry:
`ValuePins::Report` is the default, so **A1–A6 are REPORT-ONLY and the
regression tier is carried by `A7r` alone**; the default floor rule is
`resolvable` at margin 0.5, not `distinct`; and **`D-id100-negrich` shipped
2026-09-05** — `ZensimProfile::D` today loads
`d_sdr_add156_id100_negrich_dial_2026-09-05.bin` and reads **CONTRACT 6/6** and
**A7r-resolvable 5/5**. So the incumbent is not a 5/6 model with a broken
identity; it is a clean one, and this candidate has to be at least as clean
*while ranking much higher*.

### 6.1 The rows and the numbers to beat

| row | bar | shipped **D** (era-2) | the fastclass2 228 MLP |
|---|---|--:|--:|
| C1 monotonicity | ≥ 0.93 | 0.9847 canonical / **0.99470** ladder-`agree` | 0.9514 |
| C2 flat/clamp dead-zone | ≤ 0.05 | **0.0000** | 0.0017 |
| C3 negatives work | > 0 | **0.9140** | 0.3985 |
| C4 deepest probe < 0 | < 0 | **−213.149** | −70.3378 |
| C5 identity ∈ [97.5, 100] | 0 rows outside | **100.0000**, 0/38 | **90.9368, 38/38 outside — FAIL** |
| C6 no cell above identity | 0 | **0** of 4,424 | **1,642 of 9,593 (17.1 %) — FAIL** |
| **A7r** resolvable floors | 0 codecs below mentor | **0 (5/5 pass)** | **5 of 5 fail** |
| CID22 | — | 0.86338 | **0.889636** (k=3 mean) |
| W4 speed | ≤ 1.25× | — | 1.2202 max vs bar; **0.9733 max vs `zensim_D`** |

Per-codec A7r mentor bars (ladder instrument, `resolvable` @ 0.5) — a candidate
must meet **every** one: `avif-rav1e 0.6410256410256411`, `avif-svt 1.0`,
`jpeg 0.6666666666666666`, `jxl 0.9615384615384616`, `webp 1.0`. The `distinct`
bars (`0.5385 / 1.0 / 0.5385 / 0.9231 / 1.0`) are also graded, because that is
the pinned rule every published board number uses.

Two-reference inversions feed **C1 only** (there is no separate row): the
practical bar is shipped D's **0.53 %** dial-attributed rate on the ladder
instrument, against the mentor's 0.84 %. `agree` is the default; `--inversion-truth
agree` without a readable `--reference-truth` degrades to `single` **loudly**.
Unknown attribution is never an exemption — it stays charged to the dial.

### 6.2 The invocations

```sh
# rank + dial + G-ADDR, one command, the same `grade` every published number used
ZL_BV=$PWD/target/release/bake_verdict ZL_ERA=ladder \
ZL_FLOORRULE=resolvable ZL_FLOORMARGIN=0.5 ZL_TAILPINS=product \
  scripts/dialgate_arms.sh score <label> <bake.bin> 372
# also grade the pinned rule
ZL_FLOORRULE=distinct scripts/dialgate_arms.sh score <label>_distinct <bake.bin> 372
# encoder-inversion census (bake-independent evidence for anything C1 charges us)
bake_verdict --bake <bake.bin> --dial-grid $L/dial_grid_372col_ladder.parquet \
  --corpora cid22 --reference-truth $L/reference_truth_ladder_pnorm3.tsv:pnorm3 \
  --encoder-inversion-census <out>.tsv --output /dev/null
```
`$L = /mnt/v/output/zensim/ladder-2026-09-05/instruments`. Probes are the
**postC** 372-wide pair under `/mnt/v/output/zensim/dpeaks372-2026-09-05/instruments/`
(the identity probe is byte-identical to the 09-04 one — the zero vector is
era-invariant at w372).

### 6.3 The rest

1. **Selection**: `freeze_check --select --seed-group --min-k 2 --floor-basis all`.
   Known limitation carried forward: a k=1 cell can still win it; `--seed-group`
   makes k visible, it does not prevent that. An **UNMEASURED M3a cell is listed
   but not selectable**, so M3a is measured on every candidate — **on the PACKED
   bake**, exclusively (≈66 s/bake, never concurrent with fits: 36 bakes ≈ 40 min).
2. **Rank**: paired bootstrap vs shipped D **and** vs the three replicated 944
   leaders **on their own compute**, over CID22, KonJND `|·|`, AIC-3, TID, KADID,
   CSIQ, LIVE, `hfnl_cid22band`. Noise floor to respect (per-model bootstrap CI
   half-widths): **CID22 0.0066–0.0069, KonJND 0.0733–0.0745, AIC-3 0.0342–0.0349.**
   At k=3 a paired t has df = 2 and critical value **4.30**.
3. **Speed**: `scripts/kernel_fastclass_sweep.sh`-class driver — arms interleaved,
   3-char arm names so the env block is byte-identical, min-of-iters in-process,
   min over ≥15 process starts, ASLR on, **two bit-identical controls**, box-load
   self-check that SKIPS rather than emit a contaminated number. Both tiers,
   1T + 8T, 576² + 1152². The 2026-09-01 lesson is in force: a stable reference
   arm reading below a plausible floor means the harness degenerated, and `min()`
   will happily select the corrupted reading. Denominator is **`zensim_D`**, not
   `add156_156basic` (which builds `V1PoolsMode::Off`, a mode production never
   selects, and is *slower* than production).
   A 228-slice bake serves through `V1PoolsMode::Peaks` — the mode `ZensimProfile::D`
   already resolves to — so its walk delta vs D is ~0 by construction, and the
   MLP forward is **below this instrument's noise floor** (measured: extract-only
   arms read *slower* than their full siblings).
4. **Corruption**: the ZCTH tree head via `bake_verdict --corruption-head`; refit
   on rev2 features if the selected candidate is rev2.
5. **Servability census**; `zentrain.feature_ids` + `feature_set_id` stamped
   (`basic+peaks@w372/…#3fb78648` — the id is layout-independent and already
   registered).

**Ship rule.** Install into `ZensimProfile::D` **only** on a full pass with
CID22 ≥ today's D (**0.86338**) CI-clean, **CONTRACT 6/6**, **A7r 5/5**, and no
regression axis lost — following `benchmarks/d_ship_flip_2026-09-05.md` §6.
Otherwise **PROPOSE** with the complete table. A partial pass is never a ship.

**Reported regardless of outcome**: the constraint cost —
`rank(nonneg+ladder) − rank(plain)` at matched seeds, capacity and features.

---

## 7. DEAD — pre-registered as out of scope, with the number that killed each

Nothing on this list is attempted in this lane. If a result appears to demand
one, that is a finding to write down, not a licence to start.

| dead | why |
|---|---|
| capacity sweeps | MEASURED: H32 ≈ H128 on the 228 set (fastclass2). The H32/H128 axis here is a confound control, not a search. |
| KonJND data-mass levers | MEASURED dead (fastclass distill wave); the CID22 cost of the data-mass trade was certified in the wave-7 KonJND work. |
| KonJND teacher / distillation levers | dead for this axis (distill wave), and the `predict --ensemble` raw-units defect means pre-`58baf010` teacher tables are not a valid base anyway. |
| dial masks | a mask hides a weights defect; this whole plan exists because that class does not hold. |
| per-class feature definitions | feature ids are append-only and era-stamped; a per-class definition is an era break with no owner. |
| spline / anchor repair of a weights defect | gate doc §10.3: C2 ⊻ C6 is unsatisfiable by any monotone spline. This is the refuted approach the plan replaces. |
| retraining the 944 leaders | out of lane; they are compared on their own compute, not re-fit. |
| touching AVIF | user HOLD (backend rewrite) is in force. |

---

## 8. Budgets and stop conditions

- Phase A: owner fixes + tests, local, minutes-scale builds.
- Phase B: 36 fits at roughly 8 min each, serialized ⇒ ~5 h of grinding, plus
  inline harvest. Local only.
- Phase C: gates are rescores over stored tables plus one idle-box W4 window.
- **Honest stop**: if a gate cannot be run because an input does not exist, it is
  reported **NOT MEASURED** with the reason — never a zero, never a pass, never
  quietly dropped. If the thesis is refuted by the constraint cost, that is the
  deliverable.

---

## RESULTS

Full record: [`benchmarks/best_of_all_2026-09-06.md`](../benchmarks/best_of_all_2026-09-06.md).
**27 cells (9 arms × k=3), 0 failures.**

**DECISION: PROPOSE, do not install.** The ship rule fails on **two independent
clauses** — A7r 5/5 fail, and KonJND is a CI-clean regression (−0.06219, interval
excludes zero). CID22 (+0.020 CI-clean vs shipped D) and CONTRACT (6/6 on every
seed) pass. Two of four is not a ship. `zensim/weights/` and `ZensimProfile::D`
are untouched.

**The thesis (§0) is CONFIRMED, with a price.** The dial contract is achievable
in the weights: C6 goes **1,642 → 0** on every seed *while* `tied` goes
0.0017 → 0.0000, so the gate record's C2 ⊻ C6 either/or is **dissolved, not
traded**. The pre-registered constraint cost is **−0.0091 CID22** pooled
(−0.0052 per-ref), larger than the control's own seed spread on 3 of 4 axes.

**The A7r null is NARROWER than §5.11 first said, and that is the lane's most
useful result.** A7r is unmoved by architecture, loss and hinge weight — but the
DATA-isolating A/B (`H_anchorlad` vs the band-weight-matched `D_lad20m`, differing
*only* in whether the ladder pairs reach the encoders' true floors) moves **4 of
5 codec floors**: svt **+0.188**, jpeg +0.051, jxl +0.051, webp +0.026. The one
that does not move is `avif-rav1e` — **the one codec with no anchor ladders**,
exactly as §9 pre-registered before any fit. Direction right, quantity
insufficient at 32 references × 4 codecs.

**Hypotheses stated and their fate:**

| pre-registered | outcome |
|---|---|
| `score = pin − g(x)` makes C5/C6 structural | **CONFIRMED**, with a condition §2.4 records that this wave does not meet |
| the architecture does not buy A7r (§2.2) | **CONFIRMED** — and it makes the floors *worse* on 4 of 5 |
| the ladder hinge addresses A7r | **FALSIFIED** — it repays monotonicity and cuts seed spread 6.3×, and does not move A7r |
| A7r is a DATA gap, not a loss gap (§9) | **SUPPORTED** on the clean A/B, on every codec with data |
| the anchor set cannot move rav1e (§9) | **CONFIRMED** — rav1e is the one non-mover |
| `--identity-rows` is a provable no-op under N1 | **FALSIFIED** by §2.4's condition; the gate could not be written as specified (§5b D-4) |

**Five of my own claims failed under checking and are corrected in place**, each
with the measurement that overturned it: the unconditional `raw(identity) = pin`;
a pooled `hfnlproxy` figure labelled per-ref (the two disagree in *direction*); a
single-seed floor read that k=3 contradicted; a "clean A/B" whose band weights
differed; and a binary-parity check that could only ever fail.

---

## 9. ADDENDUM — `G/H_anchorlad`: is A7r a DATA gap rather than a loss gap? (pre-registered 2026-09-06, after the main wave, before any fit)

**The main wave's null.** Architecture, ladder supervision, hinge weight and
their combination all leave A7r at 5 of 5 (§5.11). The one apparent movement was
webp touching exactly its bar on one seed of three.

**The hypothesis, stated before running.** *A7r is unmoved by every loss because
the TRAINING ladders never reach the true floors the instrument grades.* safesyn's
q grid starts at **q5 and steps by 5** — it has no cells at the encoders' real
lowest settings, which is precisely the window `--floor-rule resolvable` reads
(the bottom `K = 3` mentor-resolvable steps). A hinge cannot supervise an
ordering it is never shown. If that is right, the fix is DATA, not loss.

**The data.** `/mnt/v/output/zensim/ladder-2026-09-05/anchor/out/ladder_anchor_372col_anchor.parquet`
— the ladder program's own anchor set, VERIFIED here rather than assumed:

| property | measured |
|---|---|
| rows / distinct cells | 4,552 / 4,520 (+32 identity rows) |
| references | **32** |
| **∩ with the 39 eval-grid references** | **0** — the instrument is never trained on |
| CID22 overlap (dHash) | nearest is **hamming 18**, against a d ≤ 10 contamination bar |
| codecs | jpeg 992 · webp 1,312 · jxl 1,320 · avif-svt 896 · identity 32 |
| ladder depth | **28 / 31 / 41 / 45 steps** per `(ref, codec)` vs safesyn's 16 |
| settings | the encoders' TRUE lowest (jpeg q0, webp q0, jxl d=25, svt q0) |
| target | UNCLAMPED ssim2, min **−69.97**, 404 negative rows |
| features | 372 layout, already extracted — no new extraction |

**⚠ THE ANCHOR SET HAS NO `avif-rav1e`.** That is the hardest A7r codec (bar
0.6410; the whole wave's best is 0.2564). So this arm structurally CANNOT move
rav1e, and a rav1e failure afterwards is **not** evidence against the hypothesis.
Pre-registered so it cannot be read either way after the fact. The testable
codecs are **jpeg, webp, jxl and avif-svt**.

**Two data hazards, both handled before the fit.** The parquet's `human_score` is
the RAW ssim2 in `[−69.97, 100]`, not a `[0, 1]` target — loading it as-is under
`--target-scale 100` would put it on a scale **100×** every other leg and let it
dominate the loss, which is exactly the trap the fastclass2 pre-flight caught on
`cid22_train` vs `cid22_train_norm`. A normalized copy (`human_score :=
ssim2/100`) is built first. And the 32 identity rows are **excluded** from the
training group: they are zero-feature rows, this arm is about ladder data, and
leaving them in would make it a confounded `--identity-rows` test.

**The arms.** Ladder hinge at `--tv-weight 2.0` (the weight that fully repaid
monotonicity in §5.9), TV pairs = safesyn's 175,736 **plus** the anchor's
adjacent material pairs, with **band 0 = the three lowest resolvable settings per
ladder** — the same window A7r reads — up-weighted:

- `G_anchorlad` — the PLAIN net + anchor ladders.
- `H_anchorlad` — `--nonneg-distance` + anchor ladders.

k = 3 at seeds 4004/4005/4006.

**⛔ CORRECTION, made before the arms were read: `H_anchorlad` vs `D_lad20` is
NOT the clean A/B this plan first claimed.** They share the architecture and the
`--tv-weight`, but their `--tv-band-weights` differ — `D_lad20` is
`6.0,2.0,2.0,2.0` (71.3 % of its ladder gradient on the primary corpus' low-q
band) and the anchor arms are `2.0,0.7,0.7,8.0` (46.0 % there, 31.4 % on the
floor window). So the pair differs in the WEIGHTING as well as the DATA.

**`D_lad20m` is added to fix it**: `D_lad20`'s safesyn-only pairs at the anchor
arms' band weights. `H_anchorlad` vs **`D_lad20m`** isolates the ladder DATA
exactly — same architecture, same hinge weight, same band weights, differing
only in whether the pairs reach the encoders' true floors. That is the
comparison the hypothesis is tested on.

`G` is the plain-net read and has no exactly-matched partner (no plain arm ran at
`w = 2.0`), which is stated rather than glossed.

**Confound, stated up front:** adding a training group changes the group CDF and
therefore the pair-draw RNG stream, so these arms are not sample-seed-matched to
the earlier ones. The comparison is between k=3 means with their spreads, not
between paired draws.

**The floor window's gradient share, computed before the fit so a null cannot be
blamed on it afterwards.** The TV sampler draws UNIFORMLY from the pair list and
scales by the band weight, so what matters is `n × w`:

| band | contents | n | % of draws | w | **% of ladder gradient** |
|---|---|--:|--:|--:|--:|
| 0 | primary corpus, low-q | 79,533 | 38.84 | 2.0 | 46.02 |
| 1 | primary mid-q + anchor non-floor | 62,159 | 30.36 | 0.7 | 12.59 |
| 2 | primary corpus, high-q | 49,504 | 24.18 | 0.7 | 10.03 |
| **3** | **anchor FLOOR window** | **13,550** | **6.62** | **8.0** | **31.36** |

**The floor window carries 31.4 % of the ladder gradient.** If A7r does not move
under that, the failure is not "the hinge never saw the floors".

**Gates: identical to every other arm** — A7r per codec against the mentor bars,
the full contract, two-reference inversions, rank vs shipped D and vs the 944
leaders.

**Decision rule, pre-registered.** If the four testable codecs' floors move
materially toward their bars, this arm goes to the full ship gate and the null in
§5.11 is a DATA null, not a mechanism null. If they do not move, the null is
complete across **architecture + loss + data** and the record says exactly that.

