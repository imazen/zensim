# Shared native targeting instrument — September 8, 2026

Status: pilot and full AVIF development matrix complete. Spatial benefit fails
this screen; replacement-model qualification remains incomplete.

The unpublished `zensim-target::native_probe` owner now handles source/family
validation, train-only seed calibration, premeasured attained bounds, fixed
request coverage, actual 1/2/3 complete-encode search, emitted-byte verification
and work accounting. The codec adapter supplies encoding, independent decoding
and spatial map state. The new owner reuses `SeedCurve` and
`target_search_with_backend_and_bake`; it extracts the common orchestration from
the earlier JXL instrument. JXL's exact older driver remains reproducible.

The AVIF pilot used one canonical training family (origin:2010) and one separate
development validation family (origin:3311), 17 CQ knots, the exact D by-ID model,
and the preregistered Zenravif speed-6/444/8-bit/zerosum configuration. It did not
use terminal holdouts. Each training or validation bound stage produced 51
records from 119 complete encodes, including all three fixed-CQ map iterations
for neutral/active. Their calibration curves are identical because the first
encode has no prior map; only those first scores fit seeds.

Of ten requested pilot targets, three were jointly witnessed within one score
point. The 54 completed target cases preserve all six policy/budget combinations
and all three arms. With the one-training-family curve, scalar and neutral hit
all three admitted targets by two encodes; active hits all three by three
encodes. This is a wiring/cost pilot, not an estimate of performance across
images or evidence that spatial guidance helps. Sparse witnesses leave the
other requests unresolved; do not label these gaps codec-impossible.

Checks passed: four existing Rust targeting/seed tests; direct owner Clippy
with `--no-default-features --features native-probe --lib`; exact AVIF example
Clippy and release build; complete pilot execution and emitted-byte checks;
six calibration/source rejection controls before output creation; four analyzer
rejection controls (missing case, altered score, omitted map work, changed
PNG pixels). The extended analyzer reproduces the earlier 810-case JXL target
results, paired deltas and independent matched-judge results exactly. It verifies
both bound/selected encoded bytes and the decoded pixels used by the judges.
Script lint checked all 605 runnable scripts. No CI was awaited.

Full logs, source manifests, binaries and pilot data:
`/mnt/v/output/zensim/avif-native-target-2026-09-08/`. The source binary hash is in
`PROTOTYPE_BINARY.json`; execution commands are `TRAIN_SMOKE_COMMAND.json` and
`EVAL_SMOKE_COMMAND.json`, with explicit formula revision 1 and eight Rayon
threads. Each codec call is exactly one complete encode. Map comparisons are
separate from the shared outer scalar comparison and terminal decode/score;
unused final maps are counted but never described as consumed maps.

## Completed pinned 12/8-family matrix

The adapter/shared-owner Git closure is pinned and pushed. The full AVIF
[study](https://github.com/imazen/zenavif/blob/main/benchmarks/zensim_native_targeting_2026-09-08.md)
records 612 train bounds, 408 validation bounds and 504 target cases. Only
28/80 requested targets have joint witnesses, including two negative targets.
Calibrated scalar/neutral hit 28/28 within ±1 by three encodes; active hits
24/28, with p95 absolute error 2.175 versus scalar 0.849. The 52 unresolved
requests cannot be called codec-impossible from this sparse grid.

Independent judges cover all 912 validation outputs. Fixed-CQ active steering
slightly improves SSIMULACRA2 but costs 1.8–4.8% more bytes at interpolated
matched Butteraugli quality across the four content classes. Active three-shot
outputs worsen SSIMULACRA2 in all classes. These sparse comparisons require
direct confirmation; first-shot interpolation differences are sampling effects,
not map benefit (all 56 first-shot active outputs equal scalar exactly).
All 168 neutral target outputs also equal scalar exactly.

The analyzer now includes both independent judge results in its Markdown and
rejects wrong reference identities, duplicate or missing judge pairs. Three
new rejection controls pass. Every previous JSON result remains unchanged for
the AVIF matrix and the 810-case JXL regression fixture. Training/validation
commands and judges are COMPLETE; `RESULT_COMPLETE.json` pins final evidence.
The Windows copy under `~/work/zensim-validation-2026-09-08/avif-native-targeting/`
contains the compact report and reproducibility records.

Chronology correction for the next model step: the September 7
[paired H+rav1e control](cleanup_scientific_controls_2026-09-07.md) already
completed the earlier missing-codec floor-data experiment. All five floor gates
still fail. Do not repeat the September 6 proposal as if it were unperformed.
No replacement model is qualified by these instrument or controller results.
