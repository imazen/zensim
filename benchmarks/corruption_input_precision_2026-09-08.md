# Corruption input precision — September 8, 2026

The explicit ZCTH v2 precision contract fixes the previously failed cost-4
serving audit. **Accuracy remains failed; no model is qualified or promoted.**

## Change and exact evidence

The original model was trained on f32 table values promoted to f64. Pixel
features retained additional precision, which could change a tree branch and
the corruption gate. New v2 files round each declared input to f32 before
f64 standardisation. The version participates in the schema hash. Existing v1
files retain their arithmetic, and old readers reject v2 explicitly.

The canonical trainer opts in with `input_precision: "f32"`; the default
export remains v1. This replay uses one deterministic seed, 4101, and the
already registered cost-4 recipe. Every fitted numeric section is byte-identical
to the earlier failed cost-4 head: feature IDs, scaler, tree offsets/nodes and
calibration. No extraction feature or quality threshold changed.

The existing Rust pixel audit now records raw/probability outputs from the
actual served feature row as well as canonical and stored-f32 rows. All **9,036
training comparisons**, including 526 identity attempts, have exact raw,
probability and composed-score equality across all three paths. The independent
sklearn/Rust array gate passes at **zero raw ULP, zero probability delta and
zero fire disagreements on 8,213 unique rows**. Twelve training origins only;
the eight validation origins and terminal holdouts were not evaluated.

The new reader also reproduces the old v1 head's entire 8,213-row array result.
The actual old reader refuses the v2 artifact. Boundary tests distinguish
round-before-scaler from round-after-scaler, preserve v1 behavior, and reject
version/hash mismatch and unsupported versions.

## Accuracy disposition and next action

The eight fit origins have zero honest lowering and detect/order every positive.
The four separate calibration origins measure:

| Check | Result |
|---|---|
| Corruption detection | 2,523 / 2,546 = 99.0966% |
| Real-bug detection | 150 / 155 = 96.7742% |
| Strict below-native-q20 ordering | 2,531 / 2,546 = 99.4108% |
| Honest outputs incorrectly lowered | 3 / 163 = 1.8405% — FAIL |

All three false positives are legitimate near-lossless JXL outputs: one photo
and two document outputs across two source families. Their base D scores are
96.72, 98.92 and 98.57. Serving precision is now separated from this learned
generalization failure. The next accuracy experiment needs broader admitted
honest source/codec coverage, particularly near-lossless behavior. Do not
repeat the cost sweep, change thresholds using validation, or call fitted-row
perfection generalization.

The current numerical release contract is at the top of
[MODEL_SELECTION_SCORECARD](../docs/MODEL_SELECTION_SCORECARD.md). Its new
engineering tolerances are registered requirements, not measured passes or
established perceptual JNDs. The original broader shipping objective remains
incomplete, including model accuracy, spatial RD, targeting and complete cost.

## Reproduction and checks

Artifacts: `/mnt/v/output/zensim/corruption-input-precision-2026-09-08/`.
`FIT_MANIFEST.json` pins source admission, original tables/pixel pairs, source
roles, base bake and newly built instruments. `fit/seed-4101/` contains the
exact head, independent parity vectors, full pixel audit, report and failed
screen. `PRECISION_RESULT.json` and `COMPATIBILITY.json` record the repair.
The artifact source snapshot, logs and manifest bind the implementation.

Run the saved recipe through `run-heavy`:

```bash
python3 scripts/v_next/train_corruption_head.py \
  --canonical-manifest /mnt/v/output/zensim/corruption-input-precision-2026-09-08/FIT_MANIFEST.json \
  --training-screen-only --out-dir /path/to/fresh-output
```

Relevant checks pass: 20 all-feature head tests, 14 minimal-feature head tests,
Rust serving integration including HDR, CI-exact root Clippy, modified nested
extractor Clippy, public API snapshot check, Python compilation and 605-script
lint. The logs also retain initial zero-test filter attempts; those are not
counted as verification. No latency qualification is claimed by the precision
or extraction tests.
