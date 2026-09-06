# Serving the nonlinear corruption head — `ZCTH` v1, the evaluator, and four gates (2026-09-06)

**Pre-registration:** [`../docs/PLAN_CORRHEAD_SERVING_2026-09-06.md`](../docs/PLAN_CORRHEAD_SERVING_2026-09-06.md)
(written and pushed at `8b77b277`, before any code existed).
**Lane:** `claude-corrserve`, jj sibling workspace `~/work/zen/zensim--corrserve`.
**Predecessor:** [`corruption_head_theories_2026-09-06.md`](corruption_head_theories_2026-09-06.md)
(`478bc28e`) — the modelling result this makes servable.
**Artifacts:** `/mnt/v/output/zensim/corruption-head-2026-09-05/theories/` (+ its
`_MANIFEST.json`) and `.../d228hgb/`.

**ERA:** everything is **rev1** (post-option-C `56bbcda2`,
`ssim_form::SHIPPED_REVISION = Rev1`), the era the theory lane measured. Nothing
was re-extracted. No bake in `zensim/weights/` was replaced, no profile changed,
no board cell moved, and the public API delta on a default build is **zero**.

This lane answers one question — *can the tree head actually be served?* — and
answers it with gates, not with an argument. It re-opens no modelling question.

---

## 1. The format decision: a new `ZCTH` v1, not a ZNPR metadata blob

Stated with the reading it rests on, because the alternative was live.

1. **`zenpredict` is frozen at the `zenanalyze-api` contract level** (workspace
   CLAUDE.md, USER DIRECTIVE 2026-07-19). A tree section in ZNPR v3 is a
   wire-format change to a frozen crate. That settles it before any engineering
   argument, and the engineering argument agrees.
2. **ZNPR's own dispatch would be wrong by default.** `zenpredict::Model` is
   layers-and-activations and every consumer that holds one calls
   `Predictor::predict{,_transformed}` — `bake_verdict`'s `score_grid_one` does
   exactly that. A tree hidden in `metadata[]` behind a plausible identity layer
   is **silently mis-scored by anything that does not know to look for it**: the
   same shape as the `--regime 944` entry in CLAUDE.md's Known Bugs, which is
   the defect class this repo has paid for most.
3. **The shapes genuinely differ**, and one difference is a trap. ZNPR's
   `zentrain.output_calibration_spline` is **PCHIP on `(f32, f32)` knots**;
   sklearn's isotonic is **piecewise-LINEAR on f64 knots with endpoint
   clipping**. Reusing the spline section would require re-fitting the
   calibration into a different function class — measuring one thing and
   shipping another.
4. **A separate magic is the cheap half of the safety.** `b"ZCTH"` makes
   head-vs-dial confusion a refusal at byte 0 instead of a plausible number.

What ZNPR got right is **copied, not re-invented**: a magic, a `u16` format
version, a `u64` schema hash over the canonical shape, a section table of
`(offset, len)`, and a declared-feature-id list so a head obeys the same dense
contract as `zensim::declared_feature_ids`.

Layout, reader and evaluation contract: `zensim/src/corruption_head.rs` (module
docs). Writer: `emit_zcth` in `scripts/v_next/train_corruption_head.py` — the
same file that owns `emit_znpr`, so the format has one writer.

## 2. The gates, all pre-registered, all PASS

`scripts/verify_corrhead_serving.sh` runs G1/G2/G3/G4 in one place. G5-G8 are
`cargo test -p zensim --features corruption-head`, `cargo public-api`, and the
`corrhead_forward` bench.

| gate | what it asserts | result |
|---|---|---|
| **G2** tree walk | `decision_function` vs sklearn, ulp | **0 ulp** on 33,591 test + 2,016 gate rows |
| **G1** probability | after sigmoid + isotonic | **max \|Δ\| 3.330669e-16** (test) / **0.0** (gate) |
| **G1** fire set | `{P > 0.9}` as an exact set equality | **0 disagreements / 35,607 rows** (19,308 + 662 fire) |
| **G3** composition | logistic head, before vs after `gate_score` | `--full-json` **byte-identical**, `e5dab5d1…` |
| **G4** end-to-end | `bake_verdict --corruption-head <tree>` | reproduces the theory lane **exactly** (below) |
| **G5** read set | `check_servable_by` + a negative control | PASS / REFUSED, slot named |
| **G6** public API | `cargo public-api`, default build | **zero delta** (1284 items) |
| **G7** runtime | ordering flips with the head attached | PASS, premise asserted |
| **G8** forward cost | zenbench, 1T | **659 ns/compare, 0.63× D's own forward** |

### 2.1 G4 in full — the number that matters

`bake_verdict --bake <shipped D> --corruption-grid <gb82_dog 2,016-row grid>
--corruption-head corrhead_hgb_theoryfit_w372.zcth`, against the theory lane's
own `t6_gate_pass.tsv`:

| quantity | theory lane (Python) | Rust `bake_verdict` |
|---|---:|---:|
| `hgb` head `pass_q20` | 1 | **1.0** |
| `hgb` head `pass_q10` | 1 | **1.0** |
| `hgb` **DEPLOY `pass_q20`** | **0.998512** | **0.9985119047619048** |
| `hgb` DEPLOY `pass_q10` | 0.994048 | 0.9940476190476191 |
| D's dial alone, `pass_q20` | 0.267857 | 0.26785714285714285 |
| D's dial alone, `pass_q10` | 0.153274 | 0.15327380952380953 |

`0.9985119047619048` is **671 of 672** — the record's 99.85 %, reproduced
through a completely different implementation of every step after the fit.

**And it settles a question the plan flagged in advance.** The theory lane
applies a rank tie-break (`+1e-9 · normalized_rank`) before thresholding; the
Rust path does not. Its docstring claimed `P > 0.9` was "unchanged to 9
decimals". The two paths landing on the same 671/672 is that claim **measured**,
not assumed.

### 2.2 G1/G2 — where the exactness is, and where it is not

The tree walk is exact arithmetic (comparisons plus f64 additions in a fixed
order), so **0 ulp was the bar, not a tolerance** — a difference there would be
a wrong bracket, a wrong child or a mis-parsed node. The 3.33e-16 that survives
is `exp`: numpy uses its own SIMD kernel, Rust's `f64::exp` calls the platform
libm. That residue moves **no** deadband decision on any of 35,607 rows, which
is the check that actually matters, so it is reported rather than chased.

The score-space (`h < deadband_score`) and probability-space (`p > deadband`)
readings of the same deadband were also checked to select the identical rows —
0 disagreements. They are the same operating point and **not** the same f64:
the flag's default is the literal `10.0`, the head computes `100·(1−0.9)` =
`9.999999999999998`. `bake_verdict` now uses the head's own baked value unless
the caller passes `--corruption-head-threshold`, and reports the value it
actually applied (it used to report the flag regardless — a reporting bug this
run exposed, fixed).

### 2.3 The one correction this lane had to make to itself

**sklearn's isotonic does not evaluate through the scipy function its source
names.** `IsotonicRegression._build_f` constructs
`scipy.interpolate.interp1d(kind="linear")`, and reading `interp1d._call_linear`
gives: leftmost `searchsorted` bracket, convex-combination evaluation. The first
Rust implementation reproduced exactly that, and its unit test asserted the
value that code produces.

Both were wrong. `interp1d.__init__` routes plain `linear` to
`_call_linear_np`, a one-line call to **`np.interp`** — *rightmost* bracket,
*slope* form. MEASURED on a real 90-knot isotonic fit over 25,092 queries
(every knot, every midpoint, uniform draws, both out-of-range ends):

| candidate | bit-exact vs `iso.predict` | max \|Δ\| |
|---|---|---:|
| `np.interp` on the clipped query | **yes** | 0.0 |
| slope form, rightmost bracket | **yes** | 0.0 |
| convex form, leftmost bracket (scipy's `_call_linear`) | no | 1.11e-16 |

The rightmost bracket is also the one with a property worth having: a query
exactly on knot `j` degenerates to `slope · 0 + ys[j]` and returns `ys[j]` with
**zero** rounding. Isotonic fits are made of plateau edges, so that is the
common case. Gated by `an_on_knot_query_returns_the_knot_value_exactly`.

**The lesson, since it generalises:** reading the constructor is not reading the
call. A dispatch inside `__init__` can hand you a different algorithm than the
method you read, and the two agree everywhere except the case your data is made
of.

## 3. The owner extension is additive, and that is PROVEN

`scripts/v_next/train_corruption_head.py` gained `emit_zcth`, a `--model`
dispatch on `--bake-out`, and `--deadband-t`. `can_bake` widened from
`name == "logistic"` to "the forms that have an exporter"; the MLP forms are
still refused loudly.

Re-running the incumbent `d228` recipe through the patched owner, at the tree
this lane started from (`9e16da2d`, i.e. **before** the determinism lane's
thread pin):

| artifact | shipped 2026-09-05 | patched owner |
|---|---|---|
| `corruption_head_d228.bin` | `da411c8c…` | **identical** |
| `corruption_head_d228_w944.bin` | `a7ad4e85…` | **identical** |
| `split.tsv` | — | **identical** |

The logistic provenance dict is deliberately **frozen** — the native-width bake
carries no `caller_width` key and no `model`/`sklearn` keys — because
`zentrain.repro` is serialized INTO the bake and any added key moves those
bytes. ZCTH is a new format with nothing to preserve, so it carries the fuller
record (argv, input sha256, split path, sklearn version, feature-set id).

### 3.1 Interaction with the determinism lane, measured rather than assumed

The BLAS/OpenMP thread pin landed mid-lane (`f761b902`) on the same file;
resolved keeping both sides. Re-measuring afterwards:

- **The logistic bake becomes `6f97b653…`** — which is *exactly* the "1 thread"
  value the theory doc's §9 table predicted, and exactly the value the
  determinism lane registered. Two lanes, independently. The shipped
  `da411c8c…` (28-thread) is no longer reproducible **by design**; that is
  their lane's intended consequence, not a regression here, and this lane's
  byte-identity result stands as taken at `9e16da2d`, where it proves what it
  was for: the exporter change is additive.
- **The ZCTH head is BYTE-IDENTICAL across the pin**, at both widths. `hgb`
  never went through the lbfgs BLAS solve, so it was never thread-dependent —
  consistent with the theory doc's own synthetic probe finding no thread-order
  sensitivity in `HistGradientBoostingClassifier`.

### 3.2 `t6` now shares the fit, and the refactor is gated

`corrhead_tests.t6` inlined its own `StandardScaler` + `make_classifier` +
`IsotonicRegression` sequence; the exporter would have been a fifth copy. It is
now one `fit_head` in `corrhead_theories.py`, called by `t6` and by the new
`export` command. **Re-running `t6` reproduces `t6_gate_pass.tsv` and
`t6_gate_samesource.tsv` byte-for-byte**, so the refactor is arithmetic-neutral
by measurement, not by inspection. (It also demonstrates the `hgb` fit is
deterministic run-to-run.)

### 3.3 Declared in advance, and it held

A *fresh* training run cannot bit-reproduce the theory lane's trees:
`HistGradientBoostingClassifier(early_stopping=True)` draws its internal
validation split with `train_test_split(random_state=0)` over the rows **in the
order they are stacked**, and the owner permutes its broad-honest block
(`rng.choice(9593, 9593, replace=False)`) while the study driver does not. Same
rows, different order, different trees. So G4 runs on the theory lane's OWN fit,
exported through the owner's writer.

The fresh-recipe run exists too (`.../d228hgb/`, 100 trees / 6,100 nodes, 202,526 B)
and is reported as a **separate** number, not as a reproduction: held-out-source
detection **95.6 %** at T = 0.9 with severe-honest FP **0.02 %**, broad-honest FP
**0.31 %**, matched-anchor FP **0.00 %**.

## 4. What the head costs to serve

`zensim-bench/benches/corrhead_forward.rs`, zenbench, `RAYON_NUM_THREADS=1`,
a fixed pre-built 372-wide row so the only thing varying is the forward.

| arm | mean | vs `profile_d_forward` (95 % CI) |
|---|---:|---|
| `corrhead_tree_forward` (ZCTH, 100 trees / 6,100 nodes) | **659.3 ns** | **−41.1 % to −32.0 %** |
| `profile_d_forward` (shipped ADD156 dial) | 1.05 µs | — |
| `corrhead_znpr_forward` (the INCUMBENT logistic head) | 1.76 µs | **+17.5 % to +64.2 %** |

A second, independent run of the first pair read 623.9 ns / 1.10 µs. Against a
576² extraction at ~5–6 ms, the head's forward is **~0.011 % of a compare**.

**The counterintuitive result, and it is the useful one: the "simple" linear
head is 2.7× more expensive to serve than the 6,100-node tree**, and is the only
one of the three that costs more than the dial it guards. The ZNPR path runs 372
feature transforms and a 372-wide dense layer through the `Predictor`; the tree
touches ~600 nodes and 228 standardisations. **The nonlinear head is not a speed
tradeoff.**

### 4.1 Why this needed its own bench, stated because it is a measurement hazard

`ssim2_speed_bar`'s in-situ `add156_plus_corrhead` arm is the right instrument
for "what does a served compare cost" and the **wrong** one for "what does the
tree walk cost": the head is a sub-percent rider on a multi-millisecond
extraction, so the marginal sits inside the arm's own spread. MEASURED at
576²/1T on a shared box, that group **degenerated to 3 rounds at CV 51–141 %**
and reported a "+10.3 ms marginal" for a sub-microsecond forward — a number that
is not a measurement. That is the same zenbench-under-a-tight-wall-budget
failure mode CLAUDE.md's §"A SECOND, LARGER 2304² noise source" records, showing
up at 576² because the box was shared.

The in-situ arm was still **extended** (`CorrHead` sniffs `ZNPR` vs `ZCTH`), so
both formats stay priceable end-to-end in one binary when the box is quiet.

**Caveat on all four numbers:** other lanes were active throughout (peak load
3–13). The dedicated bench's CV is 27–30 % and it discarded 234–409 noisy rounds
to keep 8–14. The paired *within-run* CIs are the trustworthy part; the absolute
means are ±~20 %.

## 5. Servability — the contract, checked

`CorruptionHead::check_servable_by(profile)` derives the profile's own
extraction plan (`fold_engine::score_plan`) and refuses a head reading a slot
the walk does not populate. **Attaching a head must never widen the walk.**

- `f0..f227` (basic + peaks) is **servable by `ZensimProfile::D`** — which is
  why the theory lane chose that slice.
- A head declaring `f300` is **REFUSED**, with the offending slot named. That
  negative control is what makes the gate mean something: a head reaching into
  the masked/IW block would force `V1PoolsMode::Full` and silently make D as
  expensive as B.

**And the availability half, measured, because "the plan computes it" and "the
result vector carries it" are different facts:** `ZensimProfile::D` turns
`skip_unread_pools` on by itself, and on a real compare it emits 372 features
with **basic 139/156 and peaks 72/72 populated**, zeroing only the pool block
`f228..371` (0/144). `B` populates all three. Both are now assertions
(`every_shipped_profile_emits_372_and_d_keeps_the_peaks_block`), not comments.

## 6. The runtime companion — proposed, not approved

Everything is `#[doc(hidden)]` behind the non-default `corruption-head` feature.
`Zensim::compute` does not read the head, so **no existing caller can observe a
different score**; the gate is asked for explicitly. Exact signatures are
registered for the user's sign-off in the plan doc §3 / §3.1.

G7 asserts the mechanism end to end. On a 128² synthetic pair, `ZensimProfile::D`:

| image | D's dial | with the head attached |
|---|---:|---:|
| honest 3×3 blur (an honest heavy loss) | **41.39** | 41.39 |
| `edge_duplicate_top_row` (structurally broken) | **99.06** | **0.00** |

The dial ranks the broken image 58 points **above** the honest one; the head
floors it to 0 and the ordering flips. The premise is **asserted**, so if D's
dial ever stops getting this wrong the test fails loudly rather than passing
vacuously. Its head is a **fixture** (a real ZCTH whose stump separates the two
measured rows) and the test says so — it gates the WIRING; model skill is gated
by G4.

**A property of the composition worth knowing before using it:** `gate_score`
floors a flagged row to `min(perceptual, 0)`, so it can only sort a corruption
below an anchor whose own score is **above zero**. That is true of the q20
anchors in the gate grid; it is not automatic. The G7 test had to use a milder
honest anchor for exactly this reason, and asserts `honest > 0` so the
requirement is visible rather than lucky.

## 7. What was NOT done

- **Nothing shipped.** No weights replaced, no profile changed, no board cell,
  no default-path behaviour, zero public API. The `corruption-head` feature is
  off unless asked for.
- **The public shape is not approved.** It is written into the plan doc for the
  user's decision; until then the surface is `#[doc(hidden)]`.
- **rev2 is out of scope.** Everything here is rev1. The REV2 WAVE's refit lane
  measured the head at revision 2 separately
  ([`rev2_refit_2026-09-06.md`](rev2_refit_2026-09-06.md) §8) and found the
  `hgb` candidate **invariant** across the flip.
- **No MLP wire format.** `mlp32`/`mlp64_32` still have no exporter and
  `--bake-out` still refuses them loudly.
- **`--full-json` gained no new key.** `corruption_head` / `corruption_deploy`
  are the existing blocks; only `threshold`'s VALUE can now differ (it reports
  what was applied).
- **The `bake_verdict` markdown title changed** — the head section now names the
  head kind (`ZNPR linear` / `ZCTH tree ensemble`). `--full-json` is unaffected;
  G3 measures exactly that.
- **The forward cost is not a clean-box measurement.** §4.1 says how dirty.

## 8. Repro

```sh
# 1. export the theory lane's own fit through the owner's writer (~30 s)
cd ~/work/zen/zensim/scripts/v_next
run-heavy --mem 40G --jobs 8 -- python3 corrhead_theories.py export

# 2. all four data-dependent gates
cd ~/work/zen/zensim
cargo build --release -p zensim-validate --bin bake_verdict --bin corrhead_parity
scripts/verify_corrhead_serving.sh            # G1 G2 G4 (+ G3 with ZL_BV_BASE)

# 3. the in-crate gates
cargo test -p zensim --features corruption-head corruption_head   # G5 G7 (18 tests)
cargo public-api -p zensim --simplified                            # G6

# 4. forward cost (do this on a QUIET box; see section 4.1)
cd zensim-bench
ZEN_HY_ADD=../zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin \
ZEN_HY_CORRHEAD=/mnt/v/output/zensim/corruption-head-2026-09-05/theories/corrhead_hgb_theoryfit_w372.zcth \
RAYON_NUM_THREADS=1 ZEN_CH_ROUNDS=2000 ZEN_CH_WALL_S=420 \
  cargo bench --bench corrhead_forward
```

G3 needs a `bake_verdict` built from the pre-change commit; pass it as
`ZL_BV_BASE`. Without it the gate prints **NOT RUN**, which is not a pass.

Artifacts: `corrhead_hgb_theoryfit_w{372,944}.zcth` (201,536 B each, 100 trees /
6,100 nodes, schemas `0x26b7d3c14ffa79fa` / `0xe8732788205bedb0`) and
`parity_hgb.npz` (66 MB: the frozen test fold + the whole gate grid, with
sklearn's `decision_function` and calibrated probability for each).
