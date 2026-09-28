# Aligned bin folding — September 14, 2026

Registered before implementation/measurement, following the hardware profile
in [the rejected row experiment](moment_rows_2026-09-14.md). Bin folding costs
20.65% sampled self cycles in the shipping 4MP finite-map diagnostic. Test a
private fast path in the existing `BinAccum` owner for scale footprints that
fit inside aligned bins (`bin` divisible by scale factor). Fold coarse rows
by bin runs, avoiding per-pixel division/intersection loops. Keep factor-one
and nonaligned paths unchanged. Preserve per-bin accumulation order, zero
handling, multiplication order and clipped right/bottom edge contributions.
No feature arithmetic, model, public API, map meaning or default changes.

Compare both f32/f64 producers with an independent source-footprint/bin-overlap
reference over dyadic and non-dyadic factors, aligned/nonaligned bins, tiny/odd/
padded geometries, signed values, cancellation and subnormals. Replay all forty
existing finite-moment TRAIN reports at bins 1/8 exactly (1,176 pixel repairs),
using the frozen four ensembles and five admitted JXL pairs. Run the existing
attribution, finite-moment/session, SIMD-permutation, HDR-retention and minimal
feature checks plus Clippy/hygiene. No new training/calibration/data admission,
EVAL or TEST access. Any scientific output change blocks acceptance.

Use the existing complete prepared-call benchmark: four ensembles, correction
off/on, 1024 squared / 2048 squared, one pinned worker, thirty accepted one-call
rounds per arm/geometry. Bracket the new binary with runs of the saved shipping
binary. Register acceptance as >=5% median improvement for enabled basic228/H32
at 4MP versus both old runs, and no >2% median regression in any disabled-map
control versus either old run. Report all arms, failures and dispersion.
No p95 qualification from summary-only output. Measure the existing bin8 process
RSS control (basic228/H32, 30 calls, both sizes, off/on). Stop if correctness or
cost admission fails; preserve the rejected candidate and results without
introducing a runtime toggle. Larger quality/targeting/corruption/HDR/native-RD
qualification remains required; no model qualifies from this implementation test.

Artifacts: `~/work/zensim-validation-2026-09-14/aligned-bins/`.

## Outcome: accept the implementation optimization

The private fast path replaces repeated bin-intersection calculations only
when each coarse footprint fits inside one bin. It preserves the original
per-bin addition chains and multiplication order, including partial right/
bottom footprints and underflow/zero handling. The full-resolution and
nonaligned fallbacks retain their original implementation. No allocations,
feature definitions, model bytes, public API or runtime flag are added.

An independent source-footprint/output-bin intersection reference agrees
bitwise in 448 folds across f32/f64 inputs, aligned/nonaligned bins, dyadic and
non-dyadic factors, tiny/odd/padded geometry, signed cancellation and subnormals.
All forty public-API TRAIN reports match the previous complete JSON exactly
(1,176 pixel repairs), including scalar results, sensitivities, density and
finite rectangle queries. The five-case H32 pass and H128 M2 failures are
unchanged. This is implementation validation, not a new quality result.

## Complete prepared-call timing

Each process interleaves all four ensembles with finite moments off/on at
1024² and 2048². Every arm has thirty accepted single-call rounds; all three
owner reliability flags are clear. Old-binary runs bracket the candidate.
These are temporally bracketed binaries, not paired samples across binaries.
A drift flag appears in the candidate 1MP output; the measured gains hold
against both bracketing controls. The three processes take 379 seconds total.

| Enabled ensemble | 1024² old-before / candidate / old-after median ms | 2048² old-before / candidate / old-after median ms |
|---|---:|---:|
| basic156 | 91.73 / 82.63 / 90.76 | 376.23 / 335.84 / 375.02 |
| basic192l8 | 109.10 / 94.79 / 108.85 | 452.22 / 394.72 / 451.58 |
| basic192max | 115.80 / 106.34 / 115.17 | 467.17 / 426.54 / 466.50 |
| basic228 | 132.09 / 117.76 / 131.51 | 544.36 / 485.68 / 543.07 |

Across both old controls, enabled median improvements range from 7.7–13.1% at
1MP and 8.6–12.7% at 4MP. Basic228/H32 improves 10.6–10.8% at 4MP, passing the
registered >=5% bar. Disabled-map changes range from a 1.9% improvement to a
0.6% regression, passing the registered <=2% regression limit in every arm.
All disabled controls, summary dispersion and mean confidence intervals remain
in [the structured results](aligned_bins_2026-09-14.results.json).

The timed call computes the complete score and map against a cached reference
and makes one rectangle query. Reference setup and isolated query latency are
not timed. Scalar scoring code is unchanged, but scalar latency is not newly
measured here. No p95 is inferred from medians or confidence intervals.
The finite-moment option remains disabled by default; this gain does not close
the much larger complete-map latency gap or qualify any model.

## Memory and checks

The separate existing RSS mode constructs thirty complete maps per process,
with one cached reference and no rectangle queries. Basic228/H32, bin 8:

| Size | Disabled peak process RSS KiB | Enabled peak process RSS KiB |
|---|---:|---:|
| 1024² | 105744 | 127204 |
| 2048² | 369732 | 454248 |

All four total-process bounds fit the worker cap on these inputs. This is not
incremental RSS, native codec memory, or a claim about other bins/models/HDR.

The broad attribution filter passes 38 library tests (two existing ignored),
the existing integration check over ten SIMD dispatch configurations, and the
revision-driver test. Three finite-moment/session tests, two HDR retained-feature
checks, minimal custom-profile build and CI-exact Clippy also pass. Public APIs
and scalar feature formulas are unchanged. No tolerance or gate was loosened.

## Broader scientific coverage remains next

A metadata-only census of the already admitted TRAIN packet confirms eleven
long-side sizes from 64 through 1024 in each partition; the five previous JXL
cases do not exhaust the available spatial evidence. Four of those same source
origins have renditions through 1024; the screenshot source reaches 384. This
read used the admitted manifest only, without new pixel/table reads, fitting,
calibration or EVAL/TEST access. Do not mistake small-case replay for native
codec qualification. Register a broader content/size/quality spatial screen,
including the stronger basic228/H128 scalar candidate and H32 control, before
further model decisions. Its source/codec/decoder identity must be pinned and
its outcomes measured through the existing public API.

The cookbook now explicitly marks its historical M2=1 observation and basic-156
map limitation as dated evidence, linking the later complete-candidate and
finite-rectangle results. September 13 split restrictions supersede the older
recipe permissions. The selected Claude memories reviewed here remain context,
not authority to reopen terminal populations or inherit a historical ship pass.
No claim is made that the full memory/document audit is finished.

Max retention and repeated extraction/map work remain measured performance
targets. Tail-latency qualification also needs the existing benchmark owner to
export samples/quantiles. Human ranking, identity ordering, corruption,
attainable-bound targeting, native spatial RD, HDR and frozen EVAL qualification
remain required. No model is qualified; the full production goal remains active.
