# `powf` MAKES THE EXTRACTOR LIBC-DEPENDENT — measured, mechanism closed, fix derived; owner + era LANDED 2026-09-06, flip NOT taken

**Found 2026-09-06 by the rev2 fleet wave's own rev1 correctness gate**, which is
the only reason anyone was looking: the gate demands `to_bits()` equality against
a stored root, and nothing weaker would have seen this.

**Status: OWNER + ERA LANDED 2026-09-06; the FLIP is still NOT taken.** The fix
moves revision-1 bytes, so it is an era break and needed the same treatment F4,
F5 and F17 got — a registered era token, a runtime-selectable owner, a measured
blast radius, and a decision recorded before the numbers exist. All four now
exist: owner `zensim/src/det_math.rs` (`RootForm` + `DetRoots`), defect
`DEFECT_F18`, era **`v1detroot`** on `FormulaRevision::Rev2`, override
`ZENSIM_ROOT_FORM=libm|sqrt`. **`ssim_form::SHIPPED_REVISION` is still `Rev1`
and `RootForm`'s default is still `LibmPowf`, so no shipped byte moves.** This
document is the measurement half; the landing half, the full feature-path
transcendental audit, the cross-libc gate and **a falsification of §3's
accuracy claim** are in
[`libc_determinism_2026-09-06.md`](libc_determinism_2026-09-06.md). Record of
the wave that found it: `zenmetrics docs/PLAN_REV2_WAVE_2026-09-06.md` §7.

---

## 1. The defect

`powf` is **not** correctly rounded, and no standard requires two libc
implementations to agree on it. The v1 extractor calls it on **144 of its 372
slots**:

| block | slots | expression | site |
|---|--:|---|---|
| basic | 36 | `(Σx⁴/n).powf(0.25)` × 3 per (scale, channel) | `feature_v2.rs:5400/5403/5406`, `streaming.rs:580/583/585` |
| peaks | 36 | `(Σx⁸/n).powf(0.125)` × 3 | `feature_v2.rs:5316-5318` |
| masked | 36 | `.powf(0.25)` × 3 | `feature_v2.rs:5320/5322/5323`, `streaming.rs:614/616/617` |
| IW | 36 | `.powf(0.25)` × 3 | `feature_v2.rs:5326/5328/5329`, `streaming.rs:623/625/626`, `iw_pool.rs:455` |

**Every other slot in the v1 path uses `sqrt`, which IEEE-754 requires to be
correctly rounded and which is a hardware instruction — so it is libc-independent
by construction.** These 144 are the whole exposure.

Consequence, stated plainly: **the same pixels produce different features
depending on which libc the binary was linked against.** The fleet links
`x86_64-unknown-linux-musl` (static, deliberately, so a worker is immune to the
base image's glibc version); every local table on the dev box was produced with
glibc.

## 2. How it was measured

**(a) Through the extractor.** The zenfleet Feature executor, same zensim source,
against the stored postC 372 root:

| build | corpus | cells | differ | worst \|Δ\| |
|---|---|--:|--:|--:|
| **musl** | csiq | 322,152 | **77** (0.0239 %) | 1.11e-16 |
| **musl** | tid | 1,116,000 | **328** (0.0294 %) | 1.11e-16 |
| **glibc** | csiq | 322,152 | **0** | — |
| **glibc** | tid | 1,116,000 | **0** | — |

Every delta is exactly one ULP at f64.

**(b) Source drift ruled out by measurement, not by argument.** Rebuilding the
musl executor with zensim pinned at **`4fbd8ff8`** — the stored root's own
recorded `build_commit` — reproduced **the same 77 cells**. So the difference is
not in the 163 commits between that and `88477e38`. The corollary is a
substantial positive result: **`4fbd8ff8` → `88477e38` moves ZERO rev1-372
feature bits**, across the whole revision-2 selector refactor, the
`feature_defs`/`Layout`/`Plan` phases, the servability work and both kernel-lane
perf commits — verified on 866 real pairs × 372 slots against a root built before
any of them.

**(c) Localised by slot pattern.** Over csiq, 66 distinct slots differ (basic 19 /
peaks 14 / masked 16 / iw 17). Inside the basic block the differing positions are
**exactly `≡ 1, 4, 7 (mod 13)`** — precisely the three `.powf(0.25)` slots, and
nothing else.

**(d) Mechanism confirmed by a probe with nothing to do with zensim.** 400,000
random doubles spanning a pooled 4th raw moment's magnitude range, `x ** 0.25`
evaluated through each libc's `pow` (CPython's `float_pow` calls libm directly):

| pair | mismatches / 400,000 |
|---|--:|
| glibc 2.43 (Ubuntu 26.04) vs glibc 2.36 (Debian bookworm) | **0** |
| glibc vs **musl** (alpine) | **276 (0.069 %)** |

First divergence: `57076.535008512925 ** 0.25` → glibc `15.456615376437254`,
musl `15.456615376437256`. That rate predicts 0.069 % × (144 ÷ 372) ≈
**0.027 %** of feature cells, against the **0.0239 %** observed. Closed.

## 2b. The fix is now GATED, and the gate passes (2026-09-06)

`scripts/verify_cross_libc_features.sh` builds one commit for both targets and
compares `to_bits()` over 220 procedurally-generated cells × 372 features:

| arm | differing feature values | of |
|---|--:|--:|
| revision 1 (`ZENSIM_ROOT_FORM=libm`) | **21** (0.0257 %) | 81,840 |
| revision 2 (`=sqrt`) | **0** | 81,840 |

0.0257 % lands inside this document's own 0.0239 % / 0.0294 % fleet-corpus
band on a completely different population. Toggling the arm on ONE binary
moves exactly **144 distinct slots**, and every slot the two libcs disagree on
is inside them. Detail: `libc_determinism_2026-09-06.md` §4.

## 3. The fix, derived rather than chosen

`x^(1/4) = sqrt(sqrt(x))` and `x^(1/8) = sqrt(sqrt(sqrt(x)))`. `sqrt` is
correctly rounded and hardware-implemented on every target this crate builds for,
so the composition is **bit-identical on every platform and every libc**, and it
is cheaper. **For these two exponents the replacement is unique; there is no arm
to select.**

> ⚠ **CORRECTED 2026-09-06.** This paragraph read "and it is *more* accurate
> than one `pow` call as well as cheaper". **The accuracy half is FALSE and was
> never measured here.** `sqrt∘sqrt` rounds TWICE where `pow` rounds once.
> MEASURED against a 60-digit `Decimal` Newton reference over 4,000 log-uniform
> doubles: the two agree on 3,455 (86.4 %); of the 545 that differ — always by
> exactly 1 ULP — **glibc's `pow` is nearer the true value in 544 and
> `sqrt∘sqrt` in 1**. On this document's own witness glibc errs by 8.8776e-16
> and the composition by 8.8860e-16. The case for the fix is **determinism and
> a bounded error**, not accuracy. Full numbers:
> `libc_determinism_2026-09-06.md` §3.

It is not free: `sqrt∘sqrt` and `powf(0.25)` differ by up to one ULP (double
rounding), so landing it moves revision-1 bytes. That is why it is registered
here instead of landed.

## 4. What a decision needed — DONE 2026-09-06, except the flip itself

* ~~An era token and an owner in the `ssim_form` / `hf_gain_form` shape~~ —
  **DONE**: `det_math::RootForm`, era `v1detroot`, `DEFECT_F18`,
  `ZENSIM_ROOT_FORM` override.
* ~~The blast radius re-measured as an arm~~ — **DONE, and the structural
  number was WRONG BY 12**: it is **156 slots**, not 144. `(M4/n)^0.25` is also
  the v2 `ssim_dev4` slot, computed in three more finalizers this document's
  four-block table did not reach. Per-cell the two arms differ by **at most 1
  ULP**, pinned in-tree.
* ~~A decision on whether it joins revision 2 or becomes revision 3~~ —
  **joins revision 2**, so there is one era boundary and one recalculation.
  The stated cost is real and is recorded in the era's own registry note: it
  invalidates any `ZENSIM_FORMULA_REV=2` table extracted before 2026-09-06, and
  `ZENSIM_ROOT_FORM=libm` is how those are reproduced.

## 5. Unmeasured, and named rather than omitted

* ~~**`metric.rs` also calls `powf`** on the score path~~ — **CONFIRMED
  2026-09-06**: the exponents are `0.5979`, `1.2244`, `0.6130` and the
  profile-supplied `b`, none of them powers of two, so **the SCORE is
  libc-dependent by the identical mechanism, on every profile**. It is
  UNFIXED — no `sqrt` composition exists for those exponents, and a polynomial
  replacement is a far larger era than a 1-ULP one. The cross-libc dump
  instrument emits the score alongside the features precisely so this stays
  measurable rather than suspected (`libc_determinism_2026-09-06.md` §4.3).
* ~~Whether any other libm call (`cbrt`, `exp`, `ln`) in the XYB path
  diverges.~~ — **RESOLVED 2026-09-06 by reading the sites, which is stronger
  than the corpus inference: there is NO libm call in the XYB path at all.**
  sRGB→linear is `linear_srgb`'s LUT / rational polynomial; the opsin cube root
  is `color::cbrtf_fast` (bit-trick seed + two Halley iterations, `mul_add`
  only) and magetypes' `cbrt_midp`; the SIMD PU-XYB path uses magetypes'
  `log2_midp_precise` / `exp2_midp_precise`. All pure IEEE arithmetic. The
  remaining libm on any feature-adjacent path is the SCALAR PU21 encode and
  `transfer.rs`'s PQ/HLG — both HDR-only, both already outside every bit-exact
  gate. Table: `libc_determinism_2026-09-06.md` §2.
