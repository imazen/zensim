# F19 — the SCORE path stops being a function of which libc linked (2026-09-06)

**Owner:** `zensim::det_math::PowForm` (the same module F18's `RootForm` lives in).
**Era:** `scorepow`, on `FormulaRevision::Rev2`. **Registered, NOT flipped** —
`ssim_form::SHIPPED_REVISION` is still `Rev1` and `PowForm::default()` is
`LibmPowf`, so no shipped byte moves.
**Gate:** `scripts/verify_cross_libc_features.sh` (`just check-cross-libc`).
**Predecessor:** `benchmarks/libc_determinism_2026-09-06.md` (F18, the FEATURE
path), whose §5 named this exposure and left it unmeasured.

---

## 1. What was wrong

`benchmarks/libm_pow_nondeterminism_2026-09-06.md` established that `powf` is
not correctly rounded and that no standard makes two libcs agree on it. F18
routed the FEATURE path's `powf(0.25)` / `powf(0.125)` through a `sqrt`
composition and closed that half.

It could not close the SCORE, and said so. `zensim/src/metric.rs` calls a libm
transcendental at **18 call sites** (10 functions) that reach a number the
product returns:

| site | call | what it is |
|---|---|---|
| `bounded_score_squash` | `d.powf(b)`, `(-x).exp()` | the shipped mapping, `100·e^{−(a/100)·d^b}` |
| `distance_to_score_mapped` | `d.powf(b)` | the legacy `100 − a·d^b` mapping |
| `ZensimResult::approx_ssim2` | `powf(0.5979)` | public API |
| `ZensimResult::approx_dssim` | `powf(1.2244)` | public API |
| `ZensimResult::approx_butteraugli` | `powf(0.6130)` | public API |
| `soft_clamp_score` | `exp` | `ProfileParams::soft_clamp_score` |
| `apply_tanh_output_pin` | `exp` | the `[0,100]` sigmoid wrap |
| per-sample-α head ×3 | `abs().powf(p)`, `powf(1/p)`, `exp` | p-norm pooling + the α gate |
| hybrid head ×3 | same | same |
| `append_mlp_size_axes` ×4 | `log2` | the four `--mlp-size-axes` MLP **inputs** |

`score_mapping_b` is **0.7 on every shipped profile** (`profile.rs`, nine sites).
`0.7` is not dyadic, so **F18's derivation cannot be reused**: there is no
finite chain of correctly-rounded operations that evaluates `x^0.7` the way
`sqrt∘sqrt` evaluates `x^(1/4)`. That is the structural difference between the
two eras — F18's arm is *derived and unique*, F19's is *chosen*.

## 2. The arm, and the two arms that were rejected on measurement

`PowForm::PureRust` is `libm::{pow, exp, log2}` — the `libm` crate, the
pure-Rust port of musl's fdlibm. **Not a new dependency**: `cargo tree -p
zensim -e normal -i libm` already showed it arriving twice (`num-traits` ←
`linear-srgb`, and `zenpredict`). Making the edge explicit is the change.

Read from `libm` 0.2.16's own source, not assumed:

* `src/math/pow.rs` and `src/math/log2.rs` contain **no
  `select_implementation!` and no `fma`** — `pow` is Sun's `e_pow.c` in
  `+ − × ÷` and bit manipulation. One source, every target, nothing to diverge
  on.
* `src/math/exp.rs` is the **one** of the three that carries a
  `select_implementation!`, gated `use_arch_required: x86_no_sse` — the x87
  path, reachable only on i586-class targets with no SSE2. `x86_64`, `i686`
  (SSE2 by default), `aarch64`, `wasm32`, macOS and Windows all take the
  portable path.

**The brief proposed reusing magetypes' `log2_midp_precise` /
`exp2_midp_precise`.** That is not available at the width this path needs, and
both halves of the rejection are measured rather than argued:

| arm | max ULP error vs a 60-digit reference | verdict |
|---|---:|---|
| `f64::powf` / `exp` / `log2` (glibc 2.43) | **1** | the shipped arm, and the defect |
| `libm::{pow, exp, log2}` | **1** | **chosen** |
| `magetypes::nostd_math::powf_f64` (f64 scalar) | **7.2e12** | rejected |
| a *perfectly rounded* f32 `powf` — a lower bound on ANY f32 route | **1.4e10**, and total loss below | rejected |

* `log2_midp_precise` & co. are defined **only** on `f32x4`/`x8`/`x16`
  (`simd/generic/generated/transcendentals_f32x*.rs`). There is no scalar form
  and no f64 form. The score path is f64 end to end.
* The f64 scalars that *do* exist are documented in magetypes' own source as
  **lowp, "~1 % max relative error"**, and `powf_f64` is literally
  `exp2_f64(n * log2_f64(x))` over those pieces. Most legible cell:
  **`log2_f64(1.0)` returns `−1.8684547677956262e−6`** where the answer is
  exactly `0`, so a 1-pixel image dimension would enter the MLP's size axes as
  `−1.9e−6`.
* Even a *perfect* f32 pow — the best case for `pow_midp_precise` — loses the
  p-norm tail outright: at the head's `p = 6`, `x = 1e−12` **underflows f32 to
  exactly `0.0`** where the true `x^6` is `1e−72`, which an f64 carries without
  effort.

Method: `zensim/examples/det_pow_probe` dumps every arm's `to_bits()` over
6,611 rows (601 log-uniform `x ∈ [1e−12, 1e3]` × the 9 score-path exponents,
801 `exp` points over `[−40, 40]`, 401 `log2` points over `[1, 1e9]`);
`scripts/det_pow_error_bound.py` prices them against `decimal` at 60
significant digits, which uses no libm at all. Split that way on purpose: the
arms must be evaluated by the shipping code, and the reference must not be a
libm.

## 3. ★ It is NOT more accurate — same correction F18 had to make

Over those 6,611 rows the two arms **disagree on 523 (7.911 %)**, and of those
the platform libm is nearer the truth on **520** and the pure-Rust port on
**3**. Both are bounded at 1 ULP.

So the case for this arm is **determinism and a measured bound**, exactly as
for `RootForm` — not accuracy. Stating it as an accuracy win would be a claim
this lane's own measurement contradicts. Pinned in-tree by
`det_math::tests::{deterministic_pow_is_within_one_ulp_of_the_truth,
both_pow_arms_are_within_one_ulp_and_libm_is_not_worse}`, which assert the
**bound** against an independently derived correctly-rounded table
(`POW_TRUTH` / `EXP_TRUTH` / `LOG2_TRUTH`, generated from the 60-digit
reference) and never against each other.

## 4. THE GATE — a 2×2, because the two defects are independent

`scripts/verify_cross_libc_features.sh` builds ONE commit twice
(`x86_64-unknown-linux-gnu`, dynamic `libm.so.6`; `x86_64-unknown-linux-musl`,
`static-pie`) and sweeps both era knobs as **runtime env vars** on the same
pair of binaries, so only one thing varies per cell. 220 procedural cells (the
20-cell parity geometry matrix + a 200-cell quantisation ladder) × 372
features, plus the score.

| `ZENSIM_ROOT_FORM` | `ZENSIM_POW_FORM` | features differing | **score differing** |
|---|---|---:|---:|
| `libm` | `libm` | **21** / 81,840 | **1** / 220 |
| `libm` | `pure` | 21 / 81,840 | **0** / 220 |
| `sqrt` | `libm` | **0** / 81,840 | **1** / 220 |
| `sqrt` | `pure` (**= revision 2**) | **0** / 81,840 | **0** / 220 |

Read the third row: **F18's fix left the score exactly as libc-dependent as it
found it** (1 → 1). Read the second: F19's fix zeroes the score while the
features are still divergent. The two defects are independent, which is why
the two forms are two env vars and not one — and this table is what *measures*
that rather than asserting it.

Four negative controls, all enforced by the script (it exits nonzero if any
fails to fire): revision 1 must show a feature difference AND a score
difference, `root=sqrt pow=libm` must show zero features, and revision 2 must
show zero on both columns. A gate that can only pass is not a gate.

## 5. What is NOT fixed — registered, not silently skipped

The audit behind these rows covered `zensim-validate` and the `zenpredict`
runtime it calls. The full classification lives in `det_math`'s module table;
the load-bearing residue:

* **`zenpredict::feature_transform` is on the PRODUCT path and is NOT
  fixable from here.** `metric.rs` calls `predict_transformed`, which reaches
  `signed_cbrt` (`cbrt`), `signed_pow` / `yeo_johnson` (`powf`, `ln`),
  `soft_clip` and the whole `log1p` family (`ln_1p`), and `Sinusoidal`
  (`sin`/`cos`). Read from the shipped bakes' own
  `zentrain.feature_transforms`: **live in Profiles A, BHdr and C**; Profiles
  **B (the default) and D are clean** — their only transform is `winsor_p99`,
  a clamp. It lives in the `zenanalyze` sibling repo, which this lane must not
  edit. `zenpredict` already has `#[cfg(not(feature = "std"))]` twins calling
  `libm::` explicitly, so the fix there is to make that the `std` path too.
* **`zensim-validate::bake_runtime` (and its `bake_compare` fork) is a
  BLOCKER on flipping `SHIPPED_REVISION`.** It re-implements the two head
  runtimes and documents itself bit-exact with `metric.rs`. That is TRUE today
  — both `PowForm` defaults are `LibmPowf` — and becomes FALSE the moment
  `scorepow` activates, because `metric.rs`'s heads now follow the form and the
  mirror does not. **No test holds them together**; the claim is prose, and a
  prior lane recorded delegation as infeasible (`bake_runtime.rs` "DEDUP-M2
  HONEST-STOP"). Routing it needs a `pub` surface on `det_math`, i.e. a
  public-API change, which is out of this lane's scope. Registered in
  `REV_SCOREPOW`'s note so a flipper reads it in the registry.
* **Verdict-only, and staying that way:** the `zenstats` panel (`logistic_eval`
  and `run_lm`'s `exp`, `phi`'s `exp`, MRR's `atanh`, `GeomeanSPP`'s `cbrt`),
  `bake_verdict`'s G3 `cbrt`, `bake_dial_refit`'s `extend-top` knot fit, and
  every trainer hit. These shape reported statistics, never a shipped score.
* **`output_calibration_spline` / `dial_spline` are already clean** — the PCHIP
  basis is `powi` only, which lowers to a multiply chain. Audited, zero
  transcendentals.
* `metric.rs:6639`'s `powf(2.4)` is inside a `#[cfg(test)]` helper.

## 6. `scorepow` moves ZERO feature slots — and that is the claim

Every other registered era answers *"which slots?"* by derivation from the
signal table. `scorepow` answers *"none"*: the score is not a feature. It is
therefore registered in `feature_defs::SCORE_PATH_REVISIONS` rather than on a
`SignalDef`, `research::era_is_registered` now consults **both** registries
(otherwise "every active era token is registered" would have broken the moment
a score era joined a revision — a registry gap, not a reason to weaken the
assertion), and `feature_defs::tests::scorepow_moves_no_feature_slot` pins the
emptiness at widths 372/720/924/944 so nobody mistakes it for an unfinished
registration, and so attaching F19 to a signal — which would invalidate stored
feature tables for a defect that cannot reach them — fails loud.

## 7. Reproduce

```sh
# the 2x2 cross-libc gate (needs the musl target)
just check-cross-libc                    # or: bash scripts/verify_cross_libc_features.sh

# the error bound
cargo run --release -p zensim --example det_pow_probe > ~/tmp/detpow.tsv
python3 scripts/det_pow_error_bound.py ~/tmp/detpow.tsv

# the pins
cargo test -p zensim --lib det_math
```

`ZENSIM_POW_FORM=libm` reproduces any pre-era score exactly; the two accepted
values are the same byte length on purpose
(`benchmarks/era2_perf_break_2026-08-31.md` §22.5).
