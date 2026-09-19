# DVIFM block-visibility — Phase-2 preregistered TRAIN screen (2026-09-19)

Protocol: `benchmarks/dvifm_screen_prereg_2026-09-19.md` (committed
`67aa9057` before any training). Hypothesis under test: *block-peak error
weighted by mutual contrast-masking visibility adds local-ordering
information that basic228 lacks.* Decision metric: per-seed paired
difference `basic228dvifm − basic228` eval signed-SROCC on the
preregistered development segment (3,125 rows; never the frozen EVAL).

**Overall verdict: NEGATIVE — no arm advanced.** Screens 2, 3 and 4 each
produced three negative paired differences (medians −0.0041, −0.0036,
−0.0023). No five-seed confirmation was run; the family stays registered
but default-OFF (`dvifm_block`) and unqualified.

## Setup and provenance

- Feature identity: `w986`, slots `f956..f985`, slots hash `685eb6ef`,
  fsid `basic+peaks+masked+iw+v2+append+append2+csfw+dvifm@w986/unknown#685eb6ef`.
- Segments (minimal-top 2026-09-13, admitted): TRAIN 8,000 rows
  (`4ad1858d…`, admission `9ba392cb…`), EVAL-dev 3,125 rows
  (`22e83f0c…`, admission `554e37d2…`), spatial manifest `65137302…`.
  Row order: `row_id` 0–7999 train, 8000–11124 eval-dev.
- Constants were derived/fitted on rows with `row_index < 8000` only.
- Block caches: `cache-lap.bin` + `cache-local.bin`, 11,125 rows each,
  18 f32 records per block, constants-independent (records are raw
  extrema; C̃ is recomputed per spec at read time). Cache→pooled-feature
  replay parity on serialized bytes: max |Δ| = 3.8e-9 (f32 narrowing).
- Disk: 20 GB under `/mnt/v/output/zensim/dvifm-screen-2026-09-19/`
  (cap 40 GB); ≥103 GB free on `/mnt/v` throughout (floor 80 GB).
- Trainer (identical across arms — arms differ only in the 30 DVIFM
  columns): h128, `withinref,both` fit group + val-only `dev` group,
  MSE w1, lr 1e-3 cosine, 8,192 pairs/epoch, stratified sampling,
  early-stop off, f32 out, final-epoch checkpoint. Init seed = seed,
  sample seed = seed+10000, identical across arms.
- Audit: `max_consumed_feature_abs_delta` ≤ 2.8e-17 on every fit
  (f32 narrowing only; served-path recompute matches extraction).

## Convergence budget

One 160-epoch probe per arm per screen round; the plateau rule compares
dev geomean3 over the last 20% vs the prior 20% (must be ≥ −0.002).

| round | basic228 | basic228dvifm | y60 | chosen E |
|---|---|---|---|---|
| screen 2 (lap) | +0.00142 | +0.00024 | +0.00055 | 40 |
| screen 3 (local) | +0.00142 | +0.00034 | +0.00055 | 40 |
| screen 4 (fitted local) | +0.00142 | −0.00023 | +0.00055 | 40 |

The basic228 and y60 probe deltas are identical across rounds — those
arms do not consume f956+, so their curves are spec-independent; the
reproduction confirms the extraction/training path is deterministic.

## Screen 1 — mechanism check: PASS

End-to-end extraction→fit→audit→report at w986 on the preregistered
subset; block cache + index emitted; consumed-feature parity exact;
cache→pool replay parity 3.8e-9.

## Screen 2 — Laplacian band, derived constants: NEGATIVE

Spec `dvifm-lap-derived.json` (`8b90ea06…`; g=1, P=1, β=0.65, ς=4,
C₀ = TRAIN p10 of min-C̃, F2 centres = TRAIN quantiles). E=40.

| seed | basic228 | basic228dvifm | paired Δ |
|---|---|---|---|
| 9201 | 0.93666 | 0.93129 | −0.00537 |
| 9207 | 0.93498 | 0.93087 | −0.00411 |
| 9211 | 0.93535 | 0.93239 | −0.00296 |

median Δ = **−0.00411** → NEGATIVE. y60 control: 0.9312/0.9296/0.9310
(below basic228 as expected).

## Screen 3 — local band, derived constants: NEGATIVE

Spec `dvifm-local-derived.json` (`f7dbaadce…`; same rule on the local
cache). E=40.

| seed | basic228 | basic228dvifm | paired Δ |
|---|---|---|---|
| 9201 | 0.93666 | 0.93306 | −0.00360 |
| 9207 | 0.93498 | 0.93294 | −0.00204 |
| 9211 | 0.93535 | 0.93132 | −0.00403 |

median Δ = **−0.00360** → NEGATIVE. Local is the less-negative band, so
screen 4 fits on the local cache.

## Screen 4 — local band, fitted constants: NEGATIVE

Fit: joint Adam (150 epochs, lr 0.03) over per-level (g, P, C₀, β, ς)
with a linear 30→1 MSE head on the local TRAIN block cache
(`row_index < 8000`); β prior λ=1e-3 toward 0.65. Loss 261.16 → 255.11.
Replay-vs-extracted parity before fitting: max |Δ| = 8.8e-8. Fitted
per-level (g, P, C₀, β, ς):

| level | g | P | C₀ | β | ς |
|---|---|---|---|---|---|
| 0 | 0.823 | 0.839 | 0.00149 | 0.604 | 3.55 |
| 1 | 0.874 | 0.927 | 0.00174 | 0.658 | 4.10 |
| 2 | 1.006 | 1.137 | 0.00194 | 0.621 | 3.70 |
| 3 | 0.992 | 1.036 | 0.00382 | 0.649 | 3.84 |
| 4 | 0.524 | 0.835 | 0.02223 | 0.617 | 4.94 |

F2 centres re-derived at the fitted g (`dvifm-local-fitted-final.json`,
`1b828398…`); constants baked into `DvifmParams::default()` for the
rescreen (`faae482c`). E=40 (probes satisfied the rule).

| seed | basic228 | basic228dvifm | paired Δ |
|---|---|---|---|
| 9201 | 0.93666 | 0.93395 | −0.00271 |
| 9207 | 0.93498 | 0.93369 | −0.00130 |
| 9211 | 0.93535 | 0.93303 | −0.00232 |

median Δ = **−0.00232** → NEGATIVE. Fitting roughly halved the deficit
versus the derived-constant local screen but never crossed zero on any
seed. Three seeds are a spread check, not a confidence interval; the
consistent sign across nine paired fits (3 seeds × 3 rounds) is the
evidence.

## Interpretation

The hypothesis is measured negative on this TRAIN-development screen at
the registered operating point: the 30 DVIFM columns do not add signed-
SROCC over basic228 under any tested configuration — first-screen
constants on either band, or a fitted parameterisation that materially
reduced the deficit. This is a screen result, not a proof the
information is absent: the fit moved the constants substantially
(L4 g→0.52) yet still lost ~0.002–0.003 SROCC per seed, and the y60
control shows the ceiling here is trainer/recipe-bound, not feature-
bound. Possible (unregistered) follow-ups would be a different screen
regime or joint training with the head; none are justified by these
numbers.

## Cost line

DVIFM extraction measures **+31.6 ms at 1024²** (Phase-1 gate) against
the 50 ms p95 bar — an `Expensive` family. Optimisation toward the bar
would only be justified by a confirmed paired gain; the best observed
median is −0.0023 SROCC, so no optimisation work is warranted.

## Evidence layout (large files under the output dir, not in git)

`/mnt/v/output/zensim/dvifm-screen-2026-09-19/`:

- `mech/run/`, `screen2-lap/run/`, `screen3-local/run/`,
  `screen4-fitted/run/` — RESULT.json, SUMMARY.json, REPORT.md,
  audits/, per-fit logs and bake spec.jsons (full argv).
- `probe/run/`, `probe3/run/`, `probe4/run/` — 160-epoch probe logs
  per arm.
- `cache-lap.bin`, `cache-local.bin` + `.index.jsonl` +
  `*-features.csv{,.manifest.json}` — block-stat caches and the paired
  extraction.
- `specs/` — all six spec JSONs (seed, derived, fitted) with
  provenance blocks; shas in the companion `.json`.
- `tools/` — `dvifm_cache.py`, `derive_constants.py`, `fit_params.py`,
  `cache_parity.py` (the fitting/replay tooling).

## What was NOT done

No frozen EVAL/test/terminal data was opened. No five-seed
confirmation (no survivor). No X/B channels, attribution caching or
remaining design ablations. No public API additions. No pushes. The
family remains registered `w986`, default-OFF, unqualified; the baked
`Default` constants record the last-tested (fitted-local) configuration
for reproducibility only.
