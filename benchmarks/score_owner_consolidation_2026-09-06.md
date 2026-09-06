# The SCORE path gets ONE owner — `zensim::score_math`, and the validate-side fork is gone (2026-09-06)

**Ledger ROUND 105** (`docs/DATASET_HISTORY.md` §3.54). **Lane:** `valunify`. **Follows:** F19 (`benchmarks/score_path_libc_determinism_2026-09-06.md`),
whose own exposure table registered this fork and called it *"a **BLOCKER** on flipping
`SHIPPED_REVISION`"*. **Rule:** CLAUDE.md "NO DUPLICATE IMPLEMENTATIONS — one owner per task"
and its single exception (a gated mirror is legitimate *only* with a bit-exact test).

---

## 0. The finding, in one measurement

`bake_verdict --full-json`, six shipped/board bakes × `cid22,kadid,tid,konjnd,aic3`, run
twice — once with `ZENSIM_POW_FORM=libm`, once with `=pure`:

| | A | board_v47wide | B | BHdr | D | board_Bwide |
|---|---|---|---|---|---|---|
| **BEFORE** (`b5a0cb70`) | identical | identical | identical | identical | identical | identical |
| **AFTER** | **differs (430 leaves)** | **differs (430 leaves)** | identical | identical | identical | identical |

Before the fix the evaluation tooling was **completely insensitive to the form the product
runtime obeys.** That is not an inference from reading the code; it is the fork, observed.

The four bakes that stay identical after the fix are *correct* to stay identical, and it was
measured rather than assumed: `psa=false hyb=false pin=None` for B / BHdr / D / board_Bwide,
so their entire score path is `out[0]` → PCHIP spline, and the PCHIP basis is `powi` only —
a multiply chain, never libm. Only A and `board_v47wide` carry
`zentrain.per_sample_alpha_head` **and** `zentrain.tanh_output_head` (`scale = 30.0`), and
only they can move.

**And nothing moved on the default arm:** all six `--full-json` outputs are **byte-identical**
to the pre-change binary's (`sha256` equal, 12/12 files across both arms).

---

## 1. What was duplicated

Five pieces of float math turn a bake's forward output into a score. Every one existed twice —
once in `zensim::metric` (the product runtime, F19-routed) and once in `zensim-validate` (the
bake-evaluation tooling, not routed) — with the bit-exactness claim carried **in prose and
nothing else**:

| arithmetic | product | validate mirror | forks |
|---|---|---|---|
| per-sample-α head (p-norm + α gate) | `metric::apply_per_sample_alpha_runtime` | `bake_runtime::apply_head_dispatch` | + `bake_compare::score_corpus` |
| hybrid head | `metric::apply_hybrid_head_runtime` | same | + `bake_compare::score_corpus` |
| tanh output pin | `metric::apply_tanh_output_pin` | `bake_runtime::apply_post_dispatch` | — |
| distance→score mapping | `metric::distance_to_score_mapped` | `apply_post`'s `mapped` arm | ×3 identical copies |
| PCHIP spline (derivs + eval) | `metric::{pchip_compute_derivs, apply_output_calibration_spline}` | `output_calibration_spline::{pchip_compute_derivs, apply}` | — |

Plus the metadata extractors and the positional scratch fill, forked a second time inside
`bake_compare.rs`, and the `POOL_STD_FLOOR = 0.0026` constant declared **three** times (twice
in `metric.rs`, once in `bake_runtime.rs`).

`bake_runtime.rs`'s own header explained why delegation was infeasible (DEDUP-M2, 2026-05-26).
**All four of its reasons are correct and still stand** — they are about the ENTRY POINTS
(`ZensimResult`, `fn() -> &'static [u8]`, the full ensemble/clamp pipeline, per-call `Predictor`
construction). None of them required copying the *arithmetic*, and copying the arithmetic is
what broke. That is the whole lesson of this lane.

---

## 2. What landed

**Owner:** new `zensim/src/score_math.rs`, `#[doc(hidden)] pub` — `per_sample_alpha_head`,
`hybrid_head`, `tanh_output_pin`, `distance_to_score_mapped`, `pchip_derivs`,
`pchip_eval_capped`, `POOL_STD_FLOOR`. Parameters arrive as **borrowed views**
(`PerSampleAlphaParams<'_>` / `HybridHeadParams<'_>`) because the two callers store them
differently — `metric.rs` in private structs, validate in public tuples — so neither has to
adopt the other's storage and neither allocates. `PowForm` is an **explicit argument**, not an
`active_pow_form()` read inside: that is `det_math`'s own documented discipline (read the
`OnceLock` once, above any loop) *and* it is what lets a test drive both arms in one process.

**`zensim::det_math` promoted to `#[doc(hidden)] pub`** — `PowForm`, `PowForm::for_revision`,
`active_pow_form`, `DetPow`. F19's table named exactly this as the fix and put it out of its own
scope. `RootForm` / `DetRoots` stay `pub(crate)`: they are the FEATURE path, and `bake_verdict`
reads stored parquet features rather than extracting, so no validate-side scorer can reach them
(confirmed — see §4).

**Deleted, in the same commit:**

- `metric.rs`: both head bodies, the tanh-pin body, the mapping body, `pchip_compute_derivs`,
  `pchip_endpoint`, the spline evaluator body, and two of the three `POOL_STD_FLOOR`s.
  −250 lines net.
- `bake_runtime.rs`: the head arithmetic and the pin. **Zero transcendentals remain in the file**
  (the three surviving `powf`/`exp` tokens are in doc comments).
- `bake_compare.rs`: its forked `PerSampleAlphaHeadDispatch`/`HybridHeadDispatch` aliases, its
  forked `extract_per_sample_alpha_head`/`extract_hybrid_head`, its inline positional fill, and
  ~90 lines of head arithmetic. **−144 lines**; `score_corpus` is now a `bake_runtime::score_row`
  loop.
- `output_calibration_spline.rs`: `pchip_compute_derivs`, `pchip_endpoint`, and the evaluator
  body. −66 lines.
- `qsweep_eval.rs` / `predict_features_with_bake.rs` / `score_pair_with_bake.rs`: three
  **byte-identical** copies of `fn apply_post` (verified line-for-line before the merge),
  replaced by one `bake_runtime::apply_post_mode`.

**No mirror was kept.** There is nothing left on the validate side that re-implements the
arithmetic, so the "gated mirror with a bit-exact test" exception does not apply. What remains
in `bake_runtime` is what DEDUP-M2 correctly identified as genuinely validate-shaped: metadata
parsing, `Predictor` + scratch reuse, the `CallerGather` policy, and the NaN short-circuits.

---

## 3. A second divergence, found by the consolidation — REAL in code, LATENT in practice

The validate-side PCHIP evaluator capped its upper **extrapolation** at 100 (the 2026-07-04
spline audit, recorded in CLAUDE.md's Resolved list) and left the **interior** segment uncapped.
The product runtime caps both. So `bake_verdict` could publish a score the shipped runtime
reports as exactly 100 — the same class of defect the 2026-07-04 audit closed, in the branch it
did not reach.

**The mechanism is not what it first looks like.** A first draft of the gate tried to build a
Hermite *overshoot* fixture and **failed its own vacuity guard** (`max 99.5` on knots
0/99/99.5): the Fritsch–Carlson derivative rule keeps the interpolant inside its bracketing
knots by construction, which is now pinned by
`score_math::tests::pchip_never_leaves_its_bracketing_knots_on_monotone_data`. The reachable
trigger is **a knot whose `y` exceeds 100** — which the wire format permits (`parse_payload`
bounds `x` strictly increasing and both coordinates finite, and bounds `y` not at all).

**MEASURED over all 49 bakes on disk** (`zensim/weights`, its `archive/`, and
`zensim-experimental/weights`): **0 declare a knot above 100.** The divergence was a loaded gun,
not a fired one — no published verdict moved — and it is closed by construction now.

**A third fact, recorded and deliberately NOT changed.** The lower branch's
`floor = ys[0] − (ys[n−1] − ys[0])` is a floor only for an *increasing* spline. On a
**decreasing** one it lands above `ys[0]` and the `.max` makes it a hard value: seven
`zensim-experimental` bakes (`v_balanced_v2/v3/v3_per_codec`, `v_compression_v2/v3/v3_per_codec`,
`zensim_b_phone_oled`) return exactly **200.0** at `x == xs[0]`, and `v02_372feat_cell5` returns
188.05. That behaviour is **identical in both implementations**, so it is not an owner
divergence; changing it would move product numbers. No shipped profile has a decreasing spline.

---

## 4. Gates

**`zensim-validate/tests/score_owner_parity.rs`** (4 tests, 10,000 rows each, no mounted corpus
— the bake is `include_bytes!` and rows are drawn from the bake's own `scaler_mean ± 3·scale`):

1. `validate_scorer_follows_the_pow_form` — **the load-bearing one.** Digests 10,000 scores'
   `to_bits()`, **re-execs the test binary** with `ZENSIM_POW_FORM=pure`, and requires the digest
   to CHANGE. `active_pow_form()` is a process-wide `OnceLock`, so the subprocess is not a
   convenience — it is the only way to see two arms at all.
   **Mutation-verified:** re-forking just the tanh pin back to `(-xc).exp()` reproduces the
   pre-fix state exactly — `b9488573a221d6cb` under **both** arms — and the test fails with that
   digest printed.
2. `post_dispatch_adapter_is_bit_identical_to_the_owner` — `score_from_network_output` must equal
   a hand-composed `score_math` head + pin + spline at the active form, by `to_bits()`, on all
   10,000 rows; and the composed value must be form-SENSITIVE, so gate 1 cannot be vacuous.
3. `spline_adapter_is_bit_identical_to_the_owner` — `ocs::apply` vs
   `score_math::pchip_eval_capped` over 10,001 points spanning both tails and every segment, plus
   `ocs::parse_payload`'s stored derivs vs `score_math::pchip_derivs`.
4. `the_pchip_interior_is_capped_like_the_product_runtime` — the §3 divergence, with a vacuity
   guard that re-derives the OLD uncapped branch and requires it to exceed 100 first.

**`zensim-validate/tests/no_score_path_libm.rs`** (2 tests) — the structural anti-refork gate,
the same shape as `no_private_iqa_stats.rs`: six named score-path files may not call
`.powf(`/`.exp(`/`.log2(`/… outside comments and string literals. `powi` is deliberately absent
(it lowers to a multiply chain and the PCHIP basis needs it). Scope is the SCORE path only and
the file list is explicit — trainers and probes legitimately own their own float math, and
`det_math`'s own table classifies `zenstats`' statistics as deliberately not routed. A second
test fails if a listed file is renamed away, so the gate cannot silently shrink.
**Mutation-verified:** re-introducing one `.exp(` fails it.

**`zensim/src/score_math.rs` unit tests** (9) — including `both_heads_respond_to_the_pow_form`
(a form-invariant head fails it), `the_two_pow_arms_stay_close_on_the_heads` (rel < 1e-12, i.e.
the arms are a rounding question and never a semantic one), and the two PCHIP property tests.

---

## 5. A measured fact worth not re-deriving: Profile A's HEAD is form-invariant

On all 10,000 fixture rows, A's per-sample-α head alone gives **bit-identical** output under both
`PowForm` arms — even though `|h|^6` disagrees on **9.80 %** of random doubles and `x^(1/6)` on
**14.07 %** (measured, 1e6 samples each; `exp` disagrees on **9.74 %**). The reason: A's hidden
vector reaches **±2.6e4**, so `alpha_logit` saturates the ±20 clamp, `α` is 1.0 to f64
resolution, and the entire `y_pool` term — the only place the p-norm enters — is multiplied by
`(1 − α) ≈ 2e-9` and annihilated. What moves A's score under the form is the **tanh pin's
`exp`**. This is why gate 1 digests the whole scored value rather than the head, and it is
asserted in-test so a fixture change that removes the sensitivity fails loudly.

Corollary for `p = 2` heads: `x^2` and `x^0.5` are libm special cases and the two arms agree on
**0/1,000,000** samples, so a hybrid/per-sample bake with `p_norm = 2` is form-invariant in its
head too. Not a defect — just a fact about which bakes this era can move.

---

## 5b. Public-API accounting — and one drift this lane did NOT create

`docs/public-api/zensim.txt` (the SUPPORTED surface) and
`docs/public-api/zensim.features.txt` have **ZERO item delta** — verified by
diffing both against `origin/main` with comment lines stripped. The two new
modules are `#[doc(hidden)]` and land in `zensim.internal.txt`.

That regenerated file carries **127 added item lines, of which 49 are this
lane's** (`det_math::*` + `score_math::*`) **and 78 are the corruption-head
lane's pre-existing staleness**: `corruption-head = []` is declared in
`origin/main`'s `zensim/Cargo.toml` and does **not** appear in
`origin/main`'s `zensim.features.txt` header, so that snapshot has been stale
since the feature landed. `zenutils-apidoc` regenerates whole files and the
header says DO NOT EDIT BY HAND, so the 78 lines cannot be held back without
producing a snapshot that disagrees with the tree. They are committed here as
the tool wrote them, and counted here so nobody mistakes 78 lines of
`corruption_head` surface for this lane's. Note `just api-doc-check` is a local
recipe — **no CI job runs it** (checked), which is why the staleness survived.

## 6. What this does NOT claim

- **It does not flip anything.** `PowForm::default()` is still `LibmPowf` and
  `SHIPPED_REVISION` is still `Rev1`. The lane removes a blocker; it does not take the flip.
- **It does not touch `zenpredict::feature_transform`** — the other item on F19's registered
  list, live in Profiles A/BHdr/C via `predict_transformed`. It is in the `zenanalyze` sibling
  repo, which this lane must not edit. Still open, still a blocker on flipping.
- **It does not change `bake_compare`'s numbers.** Its post-network dispatch stays
  `(None, None)` — no pin, no spline — which is byte-for-byte what it did before. That IS a real
  difference from `bake_verdict` (SROCC is rank-invariant under a monotone spline so the
  correlation columns agree, but Z-RMSE / PLCC / OR and the band tables do not), and it is now
  documented at the function rather than being an accident of a fork. Passing them through is a
  one-line change the day someone decides to.
- **`RootForm` was not exercised end-to-end here**, because no validate-side scorer can reach it:
  `bake_verdict` reads stored parquet features and never extracts. The verdict A/B above sets
  `ZENSIM_POW_FORM` only.

---

## 7. Reproduce

```sh
# the six-bake, two-arm verdict A/B (needs the 372 features root)
for form in libm pure; do for b in A B BHdr D; do
  ZENSIM_POW_FORM=$form bake_verdict --bake zensim/weights/<bake>.bin \
    --corpora cid22,kadid,tid,konjnd,aic3 --full-json out/$b.$form.json
done; done
cmp out/A.libm.json out/A.pure.json     # differs AFTER, identical BEFORE

# the gates
cargo test -p zensim-validate --test score_owner_parity --test no_score_path_libm
cargo test -p zensim --lib score_math
```
