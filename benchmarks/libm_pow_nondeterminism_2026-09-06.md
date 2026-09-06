# `powf` MAKES THE EXTRACTOR LIBC-DEPENDENT — measured, mechanism closed, fix derived, NOT landed

**Found 2026-09-06 by the rev2 fleet wave's own rev1 correctness gate**, which is
the only reason anyone was looking: the gate demands `to_bits()` equality against
a stored root, and nothing weaker would have seen this.

**Status: REGISTERED, NOT LANDED.** The fix moves revision-1 bytes, so it is an
era break and needs the same treatment F4, F5 and F17 got — a registered era
token, a runtime-selectable owner, a measured blast radius, and a decision
recorded before the numbers exist. This document is the measurement half.
Record of the wave that found it: `zenmetrics
docs/PLAN_REV2_WAVE_2026-09-06.md` §7.

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

## 3. The fix, derived rather than chosen

`x^(1/4) = sqrt(sqrt(x))` and `x^(1/8) = sqrt(sqrt(sqrt(x)))`. `sqrt` is
correctly rounded and hardware-implemented on every target this crate builds for,
so the composition is **bit-identical on every platform and every libc** — and it
is *more* accurate than one `pow` call as well as cheaper. **For these two
exponents the replacement is unique; there is no arm to select.**

It is not free: `sqrt∘sqrt` and `powf(0.25)` differ by up to one ULP (double
rounding), so landing it moves revision-1 bytes. That is why it is registered
here instead of landed.

## 4. What a decision needs

* An era token and an owner in the `ssim_form` / `hf_gain_form` shape, so
  revision 1 stays bit-identical and the new form is runtime-selectable.
* The blast radius re-measured **as an arm** — the structural 144 is the upper
  bound; the cells that actually move are the subset where the two roundings
  differ (~0.07 % per call).
* A decision on whether it joins revision 2 (one era boundary, but it invalidates
  the R6b lane's already-extracted rev2 tables and the fleet root built on them)
  or becomes revision 3.

## 5. Unmeasured, and named rather than omitted

* **`metric.rs` also calls `powf`** on the score path (`:338`, `:354`, `:1035`,
  `:1050`, `:1064` — the raw-distance → score mapping). If those exponents are
  not powers of two the SCORE is libc-dependent too, by the same mechanism, on
  every profile. Not measured here.
* Whether any other libm call (`cbrt`, `exp`, `ln`) in the XYB path diverges. The
  slot pattern says NO on this corpus — a per-pixel colour-transform divergence
  would move ~100 % of cells, not 0.024 % — but that is inference from one
  corpus, not a probe.
