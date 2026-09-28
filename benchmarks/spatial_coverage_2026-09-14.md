# Broader TRAIN spatial coverage — September 14, 2026

Preregistered before new pixel/model comparisons. The preceding finite-moment
and aligned-bin work proves a mechanism and reduces cost, but five small q50
cases cannot establish a spatially useful model. Compare three frozen Rev3
three-member ensembles: PS914 basic228/H128 (stronger human TRAIN rank), PP914
basic228/H32, and PT914U y60/H32 plain (cheap spatial control). No new fitting,
calibration, feature definition, ensemble weight or public API. All three use
the complete Rust BakeScorer surface with finite moments enabled and bin 8;
the instrument also computes the uncorrected baseline on the same pixels.

Retain the five previously selected TRAIN development origins (photo, report,
plot, screenshot, AI product). Register a 5-origin x 2-size-band x 3-quality
nomination grid: small longest-side <=256, large 384..1024; JXL q exactly 5,
50,90. Within each available cell choose longest-side nearest 192/1024,
then original bitstream SHA. Missing exact-quality cells remain unavailable,
not filled with a different q. This is stratified coverage, not an independent
causal estimate of resolution or q: renditions need not match across cells.
A content category represented by one source remains a limitation.

Use the largest power-of-two repair block <=short-side/4, capped at 64 and
floored at 8, for every model on a pair. This preserves small/large native-size
rectangles with enough blocks for rank diagnostics and bounds screening cost.
It does not test every codec partition or isolate block-size effects. Record
actual geometry, block count and any degenerate/nonfinite metric. Require
complete finite refinement support, M2>=.99 and M3f>=.70 per case; report every
failure. The experiment can reject candidates for wider investigation, never
qualify native encoder RD or select a shipping model from TRAIN alone.

All inputs inherit the reviewed product-TRAIN admission, original bytes and
source-family roles. Verify the admission against PREPARED, model hashes, and
original reference/bitstream bytes before decoding. Existing Rust native
decode-list output must match the previous native extraction's pixel hashes.
Check each new base score against the prior exact Rust-surface cached score
for that same model/row (absolute tolerance 1e-10 for ensemble accumulation;
no score clipping). Use independent PNG reading only to verify the native
witness pixel bytes. No EVAL/test population, new source admission or calibration
rows enter this screen. Keep all nominated dispositions and per-block records.

Reuse the existing diffmap_block_coherence binary from the accepted aligned-bin
build. Use at most two concurrent one-worker cases with explicit compute caps;
record wall time, actual pixel comparisons/maps/queries, tools and flags. This
is not a latency benchmark. Existing full human/codec TRAIN panels remain the
scalar-quality evidence and cannot be replaced with coherence or MAE alone.

Artifacts: `~/work/zensim-validation-2026-09-14/spatial-coverage/`.

## Results: the five-case result does not generalize

All 69 registered calls completed with finite, supported refinement and exact
prior native pixel witnesses. Each ensemble's base score matches its previous
Rust cached-row score within 1.43e-14 (registered tolerance 1e-10). Finite moments
preserve the instrument's scalar/features/sensitivities/old-density parity.
No EVAL, TEST or calibration rows were accessed; no model was fit or changed.

| Complete ensemble | Both gates pass | M2 failures | M3f failures | Worst M2 | Worst M3f | Human TRAIN SROCC¹ |
|---|---:|---:|---:|---:|---:|---:|
| basic228/H128 | 17/23 | 5 | 1 | 0.9763 | 0.5313 | 0.9048 |
| basic228/H32 | 17/23 | 4 | 2 | 0.9189 | 0.5195 | 0.8851 |
| y60/H32 | 23/23 | 0 | 0 | 0.9994 | 0.7166 | 0.8389 |

¹ Previous full TRAIN panels, unchanged: [product TRAIN](product_train_2026-09-14.md),
[scale frontier](product_scales_2026-09-14.md), and
[basic/peak comparison](product_peaks_2026-09-14.md). The linked panels/scatter,
not this single rank column, remain the scalar assessment. Codec labels are
same-buffer SSIMULACRA2 proxies, not independent human observations.

Basic228/H32's earlier five-case pass is therefore insufficient. Basic228/H128
has stronger human TRAIN rank but also fails six cases. Y60/H32 passes this
bounded spatial screen, with a narrow worst-case margin (.7166 versus .70),
and retains its lower scalar quality. None is a shipping selection. This is
one source per content category, not 23 independent content families.

M2 compares the base model's feature-gradient linearization against actual
served score changes. A failing M2 cannot be repaired by improving only the
spatial projection of that same linearization. M3f compares the deployable
rectangle predictor with those score changes. These are distinct mechanisms;
neither establishes native codec rate–distortion improvement.

## Every failing registered case

| Model | Content | Band | JXL q | Pixels | Block | M2 | M3f | Failed bar |
|---|---|---|---:|---|---:|---:|---:|---|
| basic228/H128 | photo | small | 50 | 192×256 | 32 | 0.9858 | 0.9108 | m2 |
| basic228/H32 | photo | small | 90 | 192×256 | 32 | 0.9189 | 0.9077 | m2 |
| basic228/H32 | report | small | 90 | 74×96 | 16 | 0.9996 | 0.5840 | m3f |
| basic228/H32 | report | large | 90 | 297×384 | 64 | 0.9675 | 0.9502 | m2 |
| basic228/H32 | plot | small | 5 | 128×128 | 32 | 0.9824 | 0.9647 | m2 |
| basic228/H128 | plot | small | 90 | 192×192 | 32 | 0.9879 | 0.9542 | m2 |
| basic228/H128 | screenshot | small | 5 | 108×192 | 16 | 0.9763 | 0.8389 | m2 |
| basic228/H128 | screenshot | small | 90 | 36×64 | 8 | 0.9989 | 0.5313 | m3f |
| basic228/H32 | screenshot | small | 90 | 36×64 | 8 | 0.9996 | 0.5195 | m3f |
| basic228/H128 | screenshot | large | 90 | 216×384 | 32 | 0.9854 | 0.9323 | m2 |
| basic228/H128 | AI product | small | 50 | 192×192 | 32 | 0.9835 | 0.8788 | m2 |
| basic228/H32 | AI product | large | 50 | 384×384 | 64 | 0.9884 | 0.9763 | m2 |

The tiny screenshot produces large magnitude errors as well as poor rank:
maximum absolute prediction errors are 10.66 score points (H128) and 11.12
(H32). The full arrays retain signed gains; some reference-block replacements
reduce a model's score. These interventions introduce boundaries and are not
a proof that every negative gain is a scalar scoring bug. No gains or scores
are clipped. All 69 original base scores are below 100; that does not erase
the broader TRAIN panel's above-identity failures for H32.

Finite moments are not a universal improvement. Relative to the instrument's
same-pixel uncorrected rectangle baseline, rank decreases in 7/23 H128,
5/23 basic228/H32 and 6/23 y60/H32 cases. H32's small report falls from .6685
to .5840. For y60's tiny q5 photo it rises from .6931 to .7166, changing that
one M3f disposition. Rank alone does not establish gain calibration.
The canonical Rust `panel` independently reproduces all 69 saved M2/M3f ranks
within 1e-12; baseline ranks use the same owner, without new pixel scoring.

## Follow-up diagnosis: finer bins do not rescue the tiny-text failures

Registered after the wider TRAIN screen, before six new calls: take every
image with a failing M3f (report row2445, screenshot row5324), retain all three
models including passing controls, and change only attribution bin8 to bin1.
Keep the original block sizes, pixels, bakes and thresholds. Exactly 210
additional repairs execute; scalar scores, actual changes and feature-space
linearizations remain exact.

| Model | Image | Bin8 M3f | Bin1 M3f | Both gates at bin1 |
|---|---|---:|---:|---|
| basic228/H128 | report | 0.7597 | 0.7597 | PASS |
| basic228/H32 | report | 0.5840 | 0.5840 | FAIL |
| y60/H32 | report | 0.9880 | 0.9880 | PASS |
| basic228/H128 | screenshot | 0.5313 | 0.5313 | FAIL |
| basic228/H32 | screenshot | 0.5195 | 0.5195 | FAIL |
| y60/H32 | screenshot | 0.8653 | 0.8653 | PASS |

All six rank values are unchanged. The registered rectangle boundaries align
with the original bins (apart from clipped image edges), so finer storage is
not a remedy for these queries. This does not test arbitrary sub-bin codec
rectangles or prove that binning never matters. Keep subsampled maps; do not
pay full-resolution storage to address this failure. Next inspect the existing
per-feature retained-prediction diagnostics on these TRAIN cases to distinguish
feature response approximation from max/nonlinear interactions. Investigate
M2 failures separately before changing a head or fitting another model.

## Coverage, cost and reproducibility

The 30 nominated cells contain 23 exact-quality pairs: 13 small and 10 large.
H128 passes 8/13 small and 9/10 large; basic228/H32 passes 9/13 and 8/10;
y60/H32 passes all. Missing cells remain:

- report, small, q5.
- report, large, q50.
- plot, large, q50.
- plot, large, q90.
- screenshot, large, q5.
- screenshot, large, q50.
- AI product, small, q5.

Requested q is an encoder setting, not a matched perceptual quality; renditions
and repair sizes differ across cells. Do not infer a causal size/quality curve
from this sparse grid. The 23-pair main run performs 4,878 repairs/rectangle
queries, 5,016 candidate pixel comparisons and 138 maps. Native decoding,
verification and calls take 21.39 seconds at two concurrent single-worker cases
under the 16GiB cap. This is orchestration wall time, not latency or p95.
The six follow-up calls add 210 repairs, 222 comparisons and twelve maps.

Original `PROTOCOL.md`, `REGISTRATION.json`, `INPUT_VERIFIED.json`,
`DECODE_VERIFIED.json`, `RESULT.json`, `RANKS.json`, `SUMMARY.json`, per-call
JSON/logs and the separate bin-diagnosis registration remain immutable in the
artifact root. `run.py`, `analyze.py`, `bin_diagnosis.py` and `gallery.py` record
the existing Rust owners and exact commands. No new scorer or statistic was
introduced. A pre-scoring registration path error used a publication manifest
with relative model paths; the authoritative fit manifest resolved it before
any new pixel comparison, without changing model hashes.

[Compact results](spatial_coverage_2026-09-14.results.json) contain every case,
missing cell, source report hash and signed magnitude diagnostic.
The existing gauntlet renderer serves [all checks and A/B repairs](http://localhost:3300/zensim/reports/spatial-coverage-2026-09-14/gallery/index.html),
with 12 failures selected initially. Date/scope, model count and block choices
now derive from the packet; absent swap examples no longer expose a broken
shortcut. The gallery labels TRAIN explicitly and includes the seven missing
cells in its scope. These are measured development checks, not new EVAL rows.

All larger scalar, corruption, native bounded targeting/RD, HDR and frozen-EVAL
requirements remain open. The full production goal stays active.
