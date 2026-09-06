# F18 — the feature extractor was LIBC-DEPENDENT; the owner, the era, and the gate

**Lane:** `claude-libcfix`, jj sibling workspace `~/work/zen/zensim--libcfix`.
**Source record (the measurement half):**
[`libm_pow_nondeterminism_2026-09-06.md`](libm_pow_nondeterminism_2026-09-06.md),
written by the rev2 fleet wave that found it.
**Status of the fix: REGISTERED as revision-2 era `v1detroot`, NOT flipped.**
`ssim_form::SHIPPED_REVISION` is still `Rev1` and `RootForm`'s default is still
`LibmPowf`, so **no shipped byte moves**.

---

## 1. What was already established, and what this lane adds

The source record measured the defect and derived the fix. It did **not** land
either. This lane:

1. **Audited every transcendental on the feature path** — not only the 144
   slots the source record's table names — and found **two site classes it
   missed** (below, §2).
2. **Gave the roots ONE owner** (`zensim/src/det_math.rs`, `RootForm` +
   `DetRoots`), on the `ssim_form` / `hf_gain_form` pattern the crate already
   uses for a revisable arithmetic form.
3. **Registered the era in the machine-readable registry** — `DEFECT_F18` and
   era `v1detroot` on `FormulaRevision::Rev2` — derived from each slot's own
   `Statistic`, so the registry's answer to "which slots does this move?"
   cannot drift away from the kernels.
4. **FALSIFIED the source record's accuracy claim** (§3).
5. **Built the cross-libc gate** and ran it (§4).
6. **Pinned the deterministic implementation bit-exactly** so a future
   toolchain or platform swap fails loud (§5).

## 2. The audit: two site classes the source record's table missed

The source record's table lists 144 v1 slots in four blocks. Re-derived here
from `Statistic::{L4, L8}` — the registry's own vocabulary — the count agrees
exactly at 144, which is the first independent confirmation of that number.
Two things it does not cover:

* **The v2 `ssim_dev4` slot.** `(M4/n)^0.25` appears in **three** more
  finalizers — `OnlineMoments::finish` (`feature_v2.rs`), the dense-block
  finalizer, and the append finalizer — which the source record's grep of the
  v1 blocks did not reach. That is **12 more slots at 944 width**, so the era
  is 156 slots, not 144.
* **`attribution.rs`'s basic-13 mirror.** `basic13_from_acc` recomputes the
  same three `L4` pools for the attribution density's sum-preservation
  identity. It is not a stored feature, but it must track the features it
  attributes or the identity stops holding under `ZENSIM_FORMULA_REV=2`.

One correction the other way: the source record's IW row cites
`iw_pool.rs:455`. That is `IwPool::l4`, which carries `#[allow(dead_code)]`
and is **not on any shipped path** — the IW block's real `L4` pooling is in
`feature_v2.rs` and `streaming.rs`, which the record's other rows already
name. Routed anyway (it is one call and free), but it is not part of the 144.

### Everything else on the feature path is ALREADY libc-independent

This is the load-bearing negative result, and it is why the gate in §4 can
reach zero rather than merely "fewer":

| stage | what it uses | verdict |
|---|---|---|
| sRGB → linear | `linear_srgb::default::srgb_u8_to_linear` — LUT / rational polynomial | **no libm** |
| opsin cube root, scalar | `color::cbrtf_fast` — bit-trick seed + 2 Halley iterations, `mul_add` only | **no libm** |
| opsin cube root, SIMD | magetypes' `cbrt_midp` | **no libm** |
| PU-XYB, SIMD (HDR) | magetypes' `log2_midp_precise` / `exp2_midp_precise` | **no libm** |
| every other v1 pool (`Mean`, `Max`, `Ratio`, `L2`) | `sqrt`, `+`, `*`, `/` | **IEEE-required, correctly rounded** |

So the exposure really is the `L4`/`L8` pools and nothing else on the SDR
feature path — which is what the source record's slot pattern (`≡ 1, 4, 7 (mod
13)` inside the basic block) already implied, now confirmed by reading the
sites rather than inferring from one corpus.

### Named, measured to be out of scope, and NOT routed

| site | call | why not |
|---|---|---|
| `pu21::pu21_encode` (scalar) | `powf(P[3])`, `powf(P[4])` | HDR-only, and it already disagrees with the SIMD path it is the reference for by ≤ 2e-3 (`simd_matches_scalar_within_band`), so it is on no bit-exact path today |
| `transfer::{pq_eotf, hlg_*}` | `powf`, `exp`, `log10` | HDR decode; no stored HDR table is under a `to_bits()` gate |
| `iw_pool` info-content weight | `log2(1 + w/σ²)` | `info_log_sigma_e_sq` **defaults to `None` and every non-test construction leaves it `None`** — off on every shipped path. Registered rather than silently changed |
| `HfGainForm::Log1pExcess` | `ln_1p` | a measurement arm; rev2's decided arm is `SaturatingExcess`, division only |
| **`metric.rs` score mapping** | `powf(0.5979)`, `powf(1.2244)`, `powf(0.6130)`, `powf(b)`, `exp` | **A REAL, UNFIXED EXPOSURE.** The exponents are not powers of two, so the SCORE is libc-dependent by the identical mechanism. The source record's §5 named it as unmeasured; it is now **measured** — see §4.3 — and still unfixed, because no `sqrt` composition exists for those exponents and a polynomial replacement is a far larger era than a 1-ULP one |

`powi` is **not** an exposure: `f64::powi` lowers to `llvm.powi`, which expands
to a multiply chain (or `compiler_rt`'s `__powidf2`, also a multiply chain). It
never reaches libm.

## 3. ★ CORRECTION: the fix is NOT more accurate — it is bounded and deterministic

The source record's §3 says the composition "is *more* accurate than one `pow`
call as well as cheaper." The second half stands. **The first is false**, and by
the obvious measurement: `sqrt∘sqrt` rounds **twice** where `pow` rounds once,
so it inherits a double-rounding error `pow` does not have.

MEASURED against a 60-digit `Decimal` Newton reference, 4,000 log-uniform
doubles over `e^±30`:

| | count |
|---|--:|
| the two agree exactly | **3,455 (86.4 %)** |
| they differ (always by **exactly 1 ULP**) | **545 (13.6 %)** |
| …of those, glibc's `pow` nearer the true value | **544** |
| …of those, `sqrt∘sqrt` nearer | **1** |

On the source record's own witness, `57076.535008512925`:

| form | value | error vs true |
|---|---|--:|
| true (60-digit) | 15.45661537643725491577634193… | — |
| glibc `pow` | 15.456615376437254 | 8.8776e-16 |
| `sqrt∘sqrt` (= the **musl** answer) | 15.456615376437256 | 8.8860e-16 |

**So the case for the fix is determinism and a *bounded* error, not accuracy:**
two correctly-rounded operations compose to a provably ≤1 ULP answer that is
identical on every platform, whereas `pow`'s error is implementation-defined
and bounded by no standard at all. Stating it as an accuracy win would be a
claim this lane's own measurement contradicts.

(Also worth recording: on that witness the deterministic form lands on **musl's**
bits, not glibc's. The era therefore moves the dev box's stored tables and
happens to move the fleet's musl output *less* — a coincidence of one value, not
a general property, and not a reason to prefer either.)

## 4. ★ THE CROSS-LIBC GATE — MEASURED, and it passes

`scripts/verify_cross_libc_features.sh` (`just check-cross-libc`) builds
`zensim/examples/libc_feature_dump` for **both** targets from THIS commit and
compares `to_bits()`. The arm is a runtime env var, not a rebuild, so the two
binaries are fixed and only the arithmetic varies.

**The grid:** the 20-cell parity geometry matrix (owned by
`tests/common/parity_cells.rs`, reached by `#[path]` so it cannot drift from
`fold_engine_parity`'s) + a **200-cell distortion ladder** (two geometries ×
100 monotonically increasing quantisation steps). Every pixel is generated
in-process from an integer PRNG, so the two builds are provably fed identical
bytes without putting a *decoder* in the comparison. 220 cells × 372 features
= **81,840 feature values**, plus 220 scores.

**Linkage, verified rather than assumed:**

```
gnu : ELF 64-bit LSB pie executable … dynamically linked …
      libm.so.6 => /usr/lib/x86_64-linux-gnu/libm.so.6
musl: ELF 64-bit LSB pie executable … static-pie linked
```

### 4.1 The result

| arm | quantity | differing | of | |
|---|---|--:|--:|---|
| `libm` (revision 1, **shipped**) | features | **21** | 81,840 | 0.0257 % — **the negative control fires** |
| `libm` | score | **1** | 220 | |
| **`sqrt`** (revision 2) | **features** | **0** | **81,840** | ★ **THE GATE** |
| `sqrt` | score | **1** | 220 | unchanged — this era does not touch the score |

**Before: 21 differing feature values. After: 0.**

The 0.0257 % rate sits inside the source record's own 0.0239 % / 0.0294 %
fleet-corpus band, on a completely different (synthetic) population — the
mechanism is a property of the arithmetic, not of a corpus.

The script contains its own negative control: **if revision 1 shows no
cross-libc difference it FAILS**, because a zero on the deterministic arm
proves nothing unless the instrument is demonstrably sensitive to what it
claims to measure.

### 4.2 The era exactly covers the exposure — measured through the extractor

Toggling `ZENSIM_ROOT_FORM` on ONE binary moves **4,097 of 81,840 rows
(5.01 %)** across **exactly 144 distinct slots** — which independently
reproduces the registry's `era_moved_slots("v1detroot", 372, …)` count through
the real walk rather than through the signal table.

**Every one of the 20 slots on which the two libcs disagree is inside those
144.** (`comm -23 libc_slots arm_slots` is empty.) So the era is neither too
narrow — nothing diverges outside it — nor is the fix speculative: it is
scoped to precisely the arithmetic that was non-deterministic.

Locally the divergences are scattered one-per-cell across the ladder
(`ladder_128x96_q005 f370`, `cell00_1153x72 f261`, …) with `f85` the only slot
hit twice, which is the shape a ~0.07 %-per-call ULP disagreement produces.

### 4.2b It is not a single-thread artifact

This repo has been burned by exactly that once — the 2026-08-30 finding that a
v1-372 masked/IW block was a function of `RAYON_NUM_THREADS`
(`benchmarks/v1_extractor_drift_2026-08-30.md`). So the gate was re-run at
`RAYON_NUM_THREADS=1` and `=8` on both binaries:

| | result |
|---|---|
| glibc, T1 vs T8 | **byte-identical** (whole dump, score included) |
| musl, T1 vs T8 | **byte-identical** |
| glibc T8 vs musl T8, features | **0 differing of 81,840** |

### 4.3 The SCORE is still libc-dependent, and the gate proves it

`cell13_255x96`'s score reads `404aae6ff0d4f2b2` on glibc and
`404aae6ff0d4f2b3` on musl — **one ULP, in BOTH arms**. That is `metric.rs`'s
raw-distance → score mapping, whose `powf` exponents (`0.5979`, `1.2244`,
`0.6130`, and the profile-supplied `b`) are not powers of two. The source
record's §5 listed this as unmeasured; it is now measured, on one cell of 220,
and it is **not fixed** — no `sqrt` composition exists for those exponents.

The dump emits the score beside the features precisely so this cannot go back
to being a suspicion.

### 4.4 The host's own libm-vs-deterministic rate

`det_math::tests::the_two_arms_differ_by_at_most_one_ulp` prints it:
**2,483 of 20,000 (12.4 %)** on glibc 2.43 over a log-uniform sweep of
`e^±14`, every one exactly 1 ULP — matching the 13.6 % measured against the
60-digit reference in §3. That is the *arm* difference; the *cross-libc*
difference is much rarer (0.026 %) because two libcs mostly agree with each
other even where both differ from the composition.

## 5. The pinning test, and why it can exist at all

`det_math::tests::deterministic_roots_are_bit_pinned_{f64,f32}` hold 24 f64 and
14 f32 inputs against **bit-exact expected `to_bits()` values**. A toolchain,
platform or libc swap that moved any of them fails the build.

**That table can exist only because the form is libm-free.** IEEE-754 requires
`sqrt` to be correctly rounded, so every value in it is a mathematical fact
about the input rather than a fact about the host. The equivalent table for
`LibmPowf` would be a table of *glibc's* answers and would fail on musl — which
is the defect, not a test bug, and is why the libm arm is gated by a **bound**
(`the_two_arms_differ_by_at_most_one_ulp`) rather than by pinned values.

The expected values were generated with Python's `math.sqrt` and then
**independently re-derived** with `Decimal.sqrt` at 60 digits rounded to nearest
double at each step: **23 of 23 non-zero inputs agree bit-for-bit on all three
composition depths.** So the table is not "whatever this box printed".

Four more gates ride with it:

* `nested_sqrt_is_exactly_the_sqrt_composition` — the form IS the composition;
  a future "optimisation" to a polynomial fails here.
* `sqrt_is_not_a_divergence_source` — the compiler's `sqrt` and magetypes'
  pure-Rust Goldschmidt `sqrtf`, which shares no code with any libm, agree to
  the bit. Without this, "we removed the libm call" is an argument; with it, it
  is a measurement.
* `negative_zero_is_positive_zero_in_both_arms` — the `x == 0.0` guard keeps a
  *sign-bit* difference out of the era's blast radius, so the era is purely a
  rounding question.
* `default_is_the_shipped_libm_form` — nothing is flipped.

## 6. What did NOT change

* `ssim_form::SHIPPED_REVISION` — still `Rev1`.
* `RootForm::default()` / `for_revision(Rev1)` — `LibmPowf`, i.e. the exact
  `powf` call that was there before, so a shipped build is byte-identical.
* Any weight file, any bake, any stored table.
* The public API — `docs/public-api/*` is unchanged. The one place this
  pressed against it is `research::FeatureProvenance::proposed_revision`,
  which is a single `Option<&str>` and therefore **under-reports the 48 slots
  that are now in two proposed eras**. Widening it to a list would be a public
  break, so the limitation is documented on the field and the complete answer
  stays `feature_defs::era_moved_slots`.

### 6.1 One honest limitation, inherited rather than introduced

`V2NewFeatureToggles::formula_revision` lets a **bake** declare its revision,
and `zensim/tests/per_bake_revision.rs` pins the coexistence property that
makes revision 2 shippable without refitting Profile C. That mechanism reaches
revision 2's `paired_global_contrast` half because that half is a *finaliser
parameter*.

`RootForm` does **not** thread that way, for the same reason
`ssim_form::active_luma_form` does not: it is read inside walks whose dispatch
does not carry the toggles. **So a bake declaring `Rev2` today gets rev2's
global-contrast arithmetic and the PROCESS's root form.** That limitation is
pre-existing — `per_bake_revision.rs`'s own module doc states it for the luma
form — and this lane inherits it rather than introducing it. Making it
per-request is the same kernel-dispatch change the luma form needs, and it
should be done for both at once: two halves of one era on two different
selectors is worse than one stated limitation.

It costs nothing today, because `SHIPPED_REVISION` is `Rev1` and both halves
therefore agree.

## 7. Migration, for whoever flips revision 2

* `v1detroot` invalidates any `ZENSIM_FORMULA_REV=2` table extracted **before**
  this commit — the R6b lane has some. `ZENSIM_ROOT_FORM=libm` reproduces those
  exactly from a post-commit binary.
* Flipping `SHIPPED_REVISION` re-extracts **156 slots** at 944 (144 at 372),
  not the 144 the source record priced. The 12 extra are the v2 `ssim_dev4`
  cells.
* The blast radius per cell is **exactly ≤ 1 ULP**, pinned in-tree by
  `the_two_arms_differ_by_at_most_one_ulp`.

## 8. Registered, NOT done

* **A musl leg in CI.** `.github/workflows/ci.yml` has `windows-11-arm`, a
  macOS Intel runner and an `i686-unknown-linux-gnu` `cross` job, but **no
  musl**. The in-crate pinning test runs everywhere and catches a moved
  deterministic value; it cannot catch a *feature-vector* divergence, because
  that needs the same commit built twice against different libcs and compared —
  which is what `scripts/verify_cross_libc_features.sh` does locally. Wiring
  that script into CI is one job (both targets are pure-Rust-linkable, no
  container needed) and is the natural next increment. Not done here.
* **The score's exposure.** `metric.rs`'s `powf(0.5979 / 1.2244 / 0.6130 / b)`
  is measured (§4.3) and unfixed. A deterministic replacement is a polynomial,
  not a `sqrt` composition, so it is a much larger era than this one and needs
  its own decision.
* **Per-request revision selection.** §6.1 — the root form and the luma form
  are both process-level `OnceLock`s; making either per-bake is the same
  kernel-dispatch change and should be done for both together.
