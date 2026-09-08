# Spline owner cleanup — 2026-09-07

New calibration goes through `bake_dial_refit`; runtime evaluation goes through
`zensim::BakeScorer`, including the output pin and spline. Four old writers and
their sole executable caller, the May V9 full pipeline, are retired. Exact
sources remain at [revision 4f57ce59](https://github.com/imazen/zensim/tree/4f57ce59b5c8/scripts/v_next)
and in the private audit archive. Their frozen TOML recipes, bakes, anchor data,
results and dated studies remain. Tracked executable caller searches in zensim
and the related zenmetrics/zenpapers/shared scripts found no surviving caller.
The small `run_cross_codec_v*_seed.sh` recipe aliases are retained.

This is archival retirement for the abandoned V9/V11 studies, not a claim that
the current quantile fitter reproduces their different band-selection method.
Historical runs require the recorded revision and instruments; do not rerun a
May command against September features and call it reproduction.

| Historical writer | Actual fitting/serialization behavior | Disposition |
|---|---|---|
| `calibrate_v9_spline.py` | May V9 372-column anchors; float32 feature pipe, six-decimal predictions; median per target band, sorted by target; greedily drops x gaps ≤1e-4. The header's epsilon-shift description disagreed with its implementation. | Archive with V9 full pipeline. |
| `calibrate_balanced_v9_spline.py` | 300-column prefix; medians sorted by prediction; direction inferred from the first two distinct knots, then inconsistent bands dropped. The `distance_shaped` argument did not determine the fit. | Archive; no executable caller. |
| `v11_ssim2/calibrate_v11_balanced_spline.py` | V11 ssim2 anchors, same direction/filter rule; could score an already calibrated input then replace its spline, fitting the wrong coordinate. | Archive; no executable caller. |
| `recal_v47_dial.py` | Strips old spline, scores pinned output, 18 percentile edges from 1–99%, per-bin medians, monotone filter, optional zero-tail dedup; u32 count plus little-endian f32 knots. | Use `add-spline --replace-existing` with explicit anchor and tail choice; the generic Rust command now honors the actual head/pin. |

The V47 fit was executed from its original AST on a frozen 128-row clamped
anchor before removal. Both tail choices reproduce all knots (f64 tolerance
1e-14) and **exact f32 payload bytes** in the Rust owner's checked-in fixture.
This proves the fitting primitive on those inputs. It does not assert entire
historical bake byte identity: the old feature pipe rounded to f32/six decimals,
while the current generic fit uses the serving surface directly. The September
4 Rust correction also preserves genuinely negative knots; the older `y<=1e-6`
filter incorrectly discarded them. That intentional correction stays.

## Runtime boundaries are a separate contract

All three old V9/V11 helpers and SciPy were executed on identical knots
`(0,0), (1,1), (2,4)` before retirement. The Rust runtime has independent
expected-value and mutation-tested owner gates; its endpoint behavior differs
intentionally from those diagnostic helpers:

| Input | Three old linear-tail helpers | SciPy polynomial extrapolation | Rust served spline |
|---|---:|---:|---:|
| −2 | 0 | 10 | 0 |
| 0.5 | 0.3125 | 0.3125 | 0.3125 |
| 1.5 | 2.1875 | 2.1875 | 2.1875 |
| 3 | 8 | 8 | 8 |
| 100 | 396 | −465398 | 100 |

The runtime's lower tail is linear with a bounded floor and its upper/interior
values cap at 100. Negative scores remain possible; the gauntlet displays them.
SciPy's extrapolated curve is not a product-score or rank-invariance oracle.
Decreasing historical splines still have the documented asymmetric floor
limitation in `output_calibration_spline.rs`; retirement does not certify them.

## Current owners and correctness fixes

- `dial_spline::fit_spline_knots`: percentile-edge, per-bin-median recipe used
  by current packing/refits; exact clamped V47 primitive check above.
- `output_calibration_spline::fit_monotone_spline`: distinct equal-count,
  upper-median training recipe with correlation-selected direction. Retained
  deliberately; it is not numerically interchangeable with percentile edges.
- `spline_payload`: shared serialization; rejects invalid/collapsed f32 knots.
- `bake_dial_refit add-spline`: fits the pre-calibration Rust surface, with
  `--replace-existing` and `--neg-tail true|false`; the old bare first-output
  forward omitted multi-head reduction and the output pin. A pin-bearing
  artifact test catches that defect and proves repeated refit does not compound
  the old spline.
- `bake_dial_refit predict --score-units`: the complete ensemble runs inside
  `BakeScorer::ensemble`, including members with different declared widths.
  Default raw mode remains explicitly a network-output diagnostic.
- `bake_dial_refit gate`: full dial scores come from the surface; its separate
  pre-calibration values measure spline-domain exposure. Both reach declared
  feature IDs instead of truncating a dense model's source table.

Invalid/empty/nonfinite anchors now produce no fit, and f32 coordinate collapse
refuses serialization. No alternate PCHIP runtime or new calibration registry
was introduced. The three-seed historical plain control uses the unchanged
`pack` recipe, separately validated in the training reproduction record.
