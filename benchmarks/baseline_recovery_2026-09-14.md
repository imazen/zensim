# Baseline recovery on improved extraction — September 14

Later user amendment: [optimize Rev3 and unify native color/HDR](../docs/REV3_OPTIMIZATION_AND_COLOR_PLAN.md).
Arithmetic may change with explicit versioning and refitting. The RGB8 cache
below remains an initial control; native color/HBD/HDR is part of the intended
product and cannot be qualified by this extraction. This supersedes the
“separate scope” wording below without changing the recovered cache's contract.

Latest user direction: recover incumbent-level quality on the faster/better
extraction algorithms, not by reverting the product to legacy extraction.
Prioritize this over new corruption, codec integration or infrastructure work.

Restore the admissible established supervision: full SafeSyn and CID22 TRAIN,
source-disjoint human TRAIN and admitted modern codec TRAIN. Use the existing
Rust feature/training/calibration/public-scoring owners. First rebuild the
missing oracle-training legs with current Rev3 features. Do not substitute
historical features or a KADID-only recipe. B/C/D remain frozen controls.

Extract full944 once for canonical training-cache reuse; candidate models will
use explicit cheaper read sets (fast local/Y and richer basic/peak regimes),
with actual served extraction/runtime measured. Full extraction during cache
construction does not imply full944 production cost. Exact final ID sets,
TRAIN family partitions and training recipe will freeze before fitting.

CID22 admission uses the existing TRAIN-only 17,611-pair/201-reference manifest;
no broad-library minus holdout scan or published/secret TEST reads. SafeSyn uses
its original bitstream manifest, not deleted decoded caches. Do not sample away
the training population to make experiments easy. Hash source manifests and
original files, retain keyed rows and current decoder/formula identity.

This first cache uses the established explicit legacy RGB8 SDR input contract
and current Rev3/sqrt feature formulas, with fresh same-buffer fast-ssim2 labels
and original oracle labels retained for comparison. This is the initial SDR
model-recovery path, not native color/HBD/HDR qualification. Color-specific
retraining is a separate scope; do not mix its features into this cache.

No model fitting or calibration until full input/extraction counts, finite
features/labels, source-family separation, sampling and train-only admission
are checked. Restore sufficient training (reference 120 epochs, 50,000 pairs
per epoch), applicable preprocessing derived on TRAIN and the existing
train-only signed-tail/identity anchor and spline owners. Do not import stale
Rev1 winsor thresholds into Rev3. At most two extraction regimes initially,
with paired replicas; no new architecture/feature sweep. Assess frozen final
compositions through public APIs, full panels/composite, dial/tails/scatter and
spatial gates. EVAL/public TEST cannot choose recipe, transforms or checkpoint.

## First recovered training leg

The complete17,611 CID22 TRAIN pairs across201 references are available and
re-extracted at current Rev3 full944 with same-buffer fast-ssim2. Extraction
completed in134.7s with zero failures; all16,624,784 feature values are finite.
Every source/decoded file hash, row key and peer pixel binding matches the
pre-extraction admission. The target column now contains verified peer scores,
not extraction placeholders. Source-reference grouping is explicit. No model
has been fitted; fitting/calibration/development family assignments are pending.

## Second recovered training leg — completion amendment

All 196,086 SafeSyn pairs across 3,218 source paths completed full944 extraction
in 4,347.6s (4,359.1s command), zero failures. Verification binds the original
manifest/admission/pairs hashes, keyed finite feature rows, source/decoded file
hashes, original CPU/GPU oracle labels and fresh same-buffer fast-ssim2 labels.
There are 11,591 negative fresh targets, range −743.8610 to 100; identities remain 100.
The reusable parquet preserves original oracle values and source-family keys.

Parquet SHA256: `6044fdc8cf4f646cda6457a9e73e54edd7d9fed57dfd95f1c47e1f4091220c68`.
Artifacts: `/var/tmp/zensim-validation-2026-09-14/baseline-recovery/`;
`CID22_VERIFIED.json` and `SAFESYN_VERIFIED.json` prove the two completed caches.
Both use Rev3 and legacy RGB8 SDR. Fit/development/calibration family admission
is pending. No model has been fitted, and no EVAL/public TEST or secret holdout
was read. Cache completion does not establish competitive model quality.
