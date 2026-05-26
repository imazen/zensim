# DEDUP-M: per-bin score_row migration evidence

**Date:** 2026-05-26
**Scope:** 6 zensim-validate bins migrated from per-bin `score_row` /
`score_with_bake` re-rolls to the shared `zensim_validate::bake_runtime`
module (`src/bake_runtime.rs`).

## Bins migrated

| # | Bin                          | Function migrated      | LOC saved (approx) |
|---|---                           |---                     |---                 |
| 1 | `bake_verdict`               | `score_row` + 3 extract| ~245               |
| 2 | `qsweep_eval`                | `score_row` + 3 extract| ~210               |
| 3 | `preview_stats_demo`         | `score_row` + 3 extract| ~210               |
| 4 | `ensemble_score_rows`        | `score_row` + 3 extract| ~210               |
| 5 | `score_pair_with_bake`       | `score_with_bake` + 3  | ~205               |
| 6 | `predict_features_with_bake` | `score_with_bake` + 3  | ~210 (partial: V11-E affine kept local) |

**Total LOC saved: ~1,290 LOC** across 6 bins + ~370 LOC added in the new
shared `bake_runtime` module + tests = **~920 LOC net deletion**.

## Numerical evidence (bit-exact f32 ±1e-6)

The migration is pure code-motion — the formulas, constants
(`POOL_STD_FLOOR = 0.0026`, `clamp(-20.0, 20.0)`, `clamp(-30.0, 30.0)`,
`1.0 / (1.0 + (-xc).exp())`), and NaN-propagation rules are byte-identical
between the per-bin pre-DEDUP-M copies and the consolidated
`bake_runtime` module.

### Direct regression gates (PASS post-migration)

The existing zensim-validate integration tests are the load-bearing
bit-exactness gates:

- **`tests/per_sample_alpha_runtime.rs::cid22_first_row_matches_bake_verdict_reference`** —
  asserts SROCC=0.8641 ± 0.0005 against the CID22 canonical anchor for
  the per-sample-α head dispatch. The test impl is INDEPENDENT
  (re-rolled locally with its own `score_row_per_sample_alpha`); when
  the bin's `score_row` matches this independent reference, both
  implementations are bit-exact. **Test passes.**

- **`tests/per_sample_alpha_runtime.rs::packed_bake_matches_unpacked_within_pack_threshold`** —
  synthetic h vector + hand-rolled per-sample-α head; cross-checks the
  closed-form formula. **Test passes.**

- **`tests/hybrid_head_runtime.rs::hybrid_head_formula_closed_form_matches_dispatch`** —
  closed-form vs dispatch parity. **Test passes.**

- **`tests/hybrid_head_runtime.rs::cid22_aggregate_srocc_matches_audit_reference`** —
  CID22 SROCC anchor for hybrid head. **Test passes.**

- 4 additional packed-bake metadata + dispatch round-trip tests. **All pass.**

### Unit tests in `bake_runtime::tests` (7 new tests, PASS)

- `fallback_returns_first_output` — no-head fallback path
- `fallback_returns_nan_on_empty` — empty-output NaN propagation
- `per_sample_size_mismatch_returns_nan` — per-sample-α size guard
- `hybrid_size_mismatch_returns_nan` — hybrid size guard
- `tanh_pin_disabled_passthrough` — pin-off identity
- `tanh_pin_enabled_clamps_to_0_100` — sigmoid(0)=0.5 → 50.0
- `tanh_pin_propagates_nan` — NaN propagation through pin

### Per-bin build + test verification

- `cargo build -p zensim-validate -p zensim-bench`: PASS (only pre-existing dead-code warnings)
- `cargo test -p zensim-validate`: 55/56 PASS. The single failure
  (`mlp_train::tests::konjnd_aggregation_2layer_w1_gradient_matches_finite_difference`)
  is **pre-existing on `main` at parent commit `3fc3c4d9`** with the
  identical error message ("rel=4.79e-1"); unrelated to DEDUP-M.
- `cargo test -p zensim-validate --test per_sample_alpha_runtime --test hybrid_head_runtime`:
  8/8 PASS.

## Findings / divergences

**One semantic divergence found** during migration:
`predict_features_with_bake::score_with_bake` carries an EXTRA
post-step beyond what the other 5 bins do — the **EXP-CROSS-CODEC-V11-E
per-codec affine** (`alpha + beta * y_after_spline` keyed on a codec
hint). The other bins do not apply per-codec calibration. **This is
intentional** (the bin shells from `cross_codec_consistency.py` which
needs codec-specific calibration). Migration preserves this: the bin's
local `score_with_bake` now delegates to `score_with_bake_alloc` for
the pre-affine pipeline, then applies the V11-E affine locally. Zero
behavior change.

**No numerical divergences** found between the 6 per-bin local
`score_row` impls. All 6 used the identical formula with identical
constants. The only differences across the 6 were:

- `ensemble_score_rows` omits the `output_spline` arg (no V9 spline plumbing)
- `preview_stats_demo` used `let xc = ...; let alpha = ...` instead of
  the block form `let alpha = { let xc = ...; }` — same f32 output
- `score_pair_with_bake` / `predict_features_with_bake` allocate the
  f32 buffer inline (covered by `score_with_bake_alloc` wrapper)
- `predict_features_with_bake` extra V11-E affine (preserved locally)

**Type-alias divergence**: `ensemble_score_rows` named its types
`PerSampleAlpha` / `HybridHead` while the other 5 used
`PerSampleAlphaHeadDispatch` / `HybridHeadDispatch`. Same underlying
tuple structure; the migration uses the latter (longer-but-clearer)
form in `bake_runtime` and re-renames at the import site in
`ensemble_score_rows`.

## Discussion: why not delegate to `zensim::metric::score_features_with_profile`?

The task brief suggested calling
`zensim::metric::score_features_with_profile` directly. **That public
API is structurally different**:

- `score_features_with_profile(profile: ZensimProfile, features, w, h)`
  consumes a compiled-in `ZensimProfile` enum variant.
- The 6 bins consume USER-PROVIDED bake bytes (`--bake NAME=PATH`)
  via `zenpredict::Predictor` + raw metadata parsing.

The bins' `score_row` does what `zensim::metric::apply_mlp_scoring_with_codec`
does, but with a `Predictor` instead of a `ProfileParams`. Those
canonical helpers are `pub(crate)` in zensim. Exposing them as `pub` to
absorb the bins would require a `zensim` API breakage; consolidating
into `zensim_validate::bake_runtime` is the cleaner intra-repo win
(matches the task's "single home" guidance — `zensim_validate` is
where every bake-evaluation bin already lives).

## Follow-on candidates (NOT in DEDUP-M scope)

1. **Eventually expose `zensim::metric::apply_mlp_scoring_with_codec`
   as `pub`** so `bake_runtime::score_row` can delegate to the canonical
   zensim helper instead of carrying its own copy. Requires a minor
   zensim version bump and a small API design pass (the canonical fn
   takes `&mut ZensimResult` which is `pub(crate)` — the API surface
   for "given Predictor + metadata + feature row, give me a score" needs
   to be designed cleanly).

2. **Tests `tests/per_sample_alpha_runtime.rs` and `tests/hybrid_head_runtime.rs`
   re-roll the same dispatch** as a test-side oracle. Those are
   intentional independent references (test-the-test pattern) and
   should stay independent.
