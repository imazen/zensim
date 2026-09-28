# Canonical corruption refit — September 8 registration

## Honest native-map coverage — after precision repair, before new fitting

The v2 cost-4 model still lowers three honest near-lossless JXL calibration
outputs. The retained native target runs contain honest outputs absent from
the scalar-only fit supplement: both neutral and active JXL/AVIF map arms.
Their bytes/pixels are not uniformly duplicates of scalar output. Reuse these
actual bitstreams before generating another codec wave.

Extend the existing build_corruption_corpus.py owner with a strict supplemental
manifest mode. Select only the eight existing fit origins, both non-scalar arms,
all 21 JXL distances and 17 AVIF CQ settings: 608 raw attempts. Pin source,
family, bounds, encoder provenance, bitstream bytes and current v2-capable Rust
extractor. Independently decode/extract/audit each pair; merge into the existing
serving input packet with fresh global keys. Deduplicate by source/pixel identity
through the canonical trainer as before. No new source admission, pixel resize,
validation encoding or change to the four calibration origins.

Keep D228 features, cost 4, HGB/scaler/calibration settings, f32 contract,
deadband and seed 4101 unchanged. Train once and run the 12 training origins
only through the complete Rust surface. Advance only if all existing five bars
pass on the unchanged calibration origins; otherwise retain the failure and
move to broader source/geometry coverage, not a threshold or cost sweep. This
experiment adds honest codec variation; it is not independent-source validation
or proof of useful spatial steering. No terminal holdout is used.

## Input-precision repair — later September 8, before implementation

The cost-4 serving audit fails because a tree trained on stored f32 inputs can
take a different branch on full-precision pixel f64 inputs. The initial D228
head also has one probability discrepancy despite identical composed scores.
Make precision executable model data, not an undocumented caller cast.

Extend the existing ZCTH reader/exporter with format version 2: identical
sections, but every declared feature rounds to IEEE f32 and widens to f64
BEFORE scaler subtraction/division/clipping. Include version 2 in the schema
descriptor. Version 1 retains its exact native-input arithmetic and existing
public FORMAT_VERSION constant; old readers refuse version 2. No new public
item or feature extraction is required. Concrete callers are the canonical
trainer, corrhead_parity and BakeScorer's pixel/cache corruption paths.

The canonical manifest opts in with input_precision="f32". Preserve legacy
default export, fitted trees/scaler/calibration, thresholds and source roles.
Allow one deterministic seed (4101) for this registered precision repair;
repeating identical HGB fits does not add evidence. Refit/re-export cost 4
with no other scientific changes and run only its 12 training origins through
the complete Rust pixel/cache audit. This diagnoses and repairs serving; it
does not reopen validation for the failed cost sweep. Test distinguishing
rounding-boundary cases, legacy behavior, raw/probability/fire consistency,
version/hash refusal and the real previously failing pixel row. Retain old
artifacts unchanged. Subsequent accuracy work must address honest coverage.

## D-regime companion — later user direction, before results

The user requests a head using D's existing feature regime. This supersedes
the proposal to expand D's extraction for the first all-372 head. Keep that
failed experiment intact. Fit a separate, explicitly declared f0..227 head:
156 basic and 72 peak features, all X/Y/B channels. `Plan::for_bake` already
materializes these for D; masked/IW f228..371 are excluded. The source tables
and parity caller remain full 372-column, revision 1, libm. No extraction
planner or serving guard changes. Tree inference still has a runtime cost.

Reuse the completed admission receipt, exact source roles, weighting, scaler,
HGB settings, three seeds, calibration and P>0.9 deadband below. Keep every
development selection bar unchanged. Train through the existing canonical
trainer and evaluate each exported ZCTH through the pinned Rust pixel/cache
audit. This is one bounded registered recipe; failures do not authorize
validation-based threshold tuning. Report channel operations separately from
the aggregate channel family, including region/severity and inert identities,
using keyed generator metadata from the already hash-bound source tables.
Compare with the prior frozen D+HGB on the same inputs. Low scalar chroma
weighting is a hypothesis, not a demonstrated sole cause of swap failures.

Artifact: `/mnt/v/output/zensim/canonical-corruption-d228-2026-09-08/`.

### Honest-cost follow-up — after the first D228 screen, before new fits

The first D228 recipe catches all tested RGB swaps, but fails honest-output
protection and below-q20 ordering. Its separate training calibration origins
also contain two wrongly lowered near-lossless JXL outputs (1214 and 6064).
On that calibration set, moving a monotone threshold above the highest honest
raw response would retain only 83.97% of corruption positives, below the
95% bar. A threshold change alone cannot satisfy both objectives there.

Bound the follow-up to three fit-cost arms: multiply honest **fit** row weights
by 4, 16 or 64 before the existing unit-mean normalization and balanced-class
factory. Keep source roles, scaler, calibration weights, features, HGB shape,
deadband and three seeds unchanged. This changes the learned boundary, with
no additional serving work. `--training-screen-only` runs Rust pixel/cache
scoring on the twelve training origins only; its parity vectors also exclude
validation. Report fit and calibration roles separately. Select an arm only
if every seed meets the same five bars on calibration, preferring smaller
honest weight if multiple arms pass. Only that selected arm may advance to
the eight validation origins. If none passes, retain the failures and stop
this cost sweep; do not inspect validation for these arms or relax a bar.

## Earlier admission and all-372 recipe (preserved)

This follows the frozen-head serving screen, before new native admission
instrument edits or fitting. No historical CV-ensemble metric can stand in for
the exported single fit. No terminal labels enter this work.

First extend `check_holdout_overlap` with a private `--native-png` mode. Reuse
zenpng, zenpixels-convert and zenresize; share the existing horizontal 9x8
comparison primitive with legacy dHash. Name the new hash era explicitly:
BT.709 encoded luma, zenresize Lanczos, gray8 descriptor. It is not asserted
bit-compatible with the legacy image/Lanczos3 pipeline. Preserve legacy mode.
Native mode requires exact nonzero train/holdout counts, fresh outputs and
complete decoding. Retain every input file SHA, dimensions and hash, with all
close pairs at distance <=16; <=10 remains a strict review flag. Never silently
skip a failed reference, automatically exclude a flag or change the threshold
after results. Include exact/resized positives, unrelated negatives and damaged
input controls before using this for source admission.

Protect CID22-49, AIC-3 full sources, AIC-4/SDR25 crops, CSIQ and LIVE, plus
UPIQ reference coverage as required by the split registry. Where an original
is not PNG, use an existing native decode owner and retain original/decoded
hashes. A missing decoder or unreviewed close pair leaves admission incomplete;
it does not grant a clean training view. Inspect crop/family membership too.

Proposed canonical head fit uses only the existing 12 train origins and their
native corpus/bitstream rows. Eight origins fit: 2010,1054,6068,6610,7066,9380,
8206,8384. Four origins calibrate: 1214,6064,9066,8462 (one per content class).
All eight existing validation origins remain evaluation-only. Deduplicate
source/pixel pairs before weighting, while retaining raw attempts and family
membership. Check actual family identities across these roles. Exact manifests,
weights, hyperparameters, seeds and selection bars must be recorded before fit.
This proposal alone is not admission or permission to tune on validation.

## Fixed fit and screen, before results

Reuse HGB factory defaults except `early_stopping=false`, `max_iter=100`,
`max_leaf_nodes=31`. All 372 canonical features, f32 table values promoted to
f64, fit-only StandardScaler and clip +/-8. Fit rows have equal total weight
per source, normalized to unit mean row weight; the factory retains balanced
class weights. Isotonic calibration uses the four calibration sources only,
with equal total weight per source. Deadband is fixed at P>0.9. Estimator seeds
4101/4103/4107; no stochastic data sampling. With early stopping disabled and
this small table, seeds may produce identical trees: do not interpret that as
source-coverage evidence or independent generalization trials.

`train_corruption_head.py --canonical-manifest` is an explicit mode with
no broad-honest/negrich defaults or random source splitting. `--prepare-only`
may produce deduplicated keyed views while admission remains incomplete; it
must not fit. Actual fitting requires a hash-bound complete content-admission
receipt. Emit with the existing ZCTH owner, run exact raw-tree/probability/fire
parity through `corrhead_parity`, then evaluate all original pixel pairs with
the already verified Rust extractor audit. Reports must identify a canonical
single head fit, not claim that it is the old frozen head or a CV ensemble.

Development selection bars: zero honest codec outputs lowered, overall honest
score lowering <=1%, overall unique corruption detection >=95%, real-bug
detection >=90%, and strict below-native-q20 ordering >=99%, on the eight
validation sources. These do not qualify the base model or spatial product.
No threshold search after the validation read; retain failed seeds. Terminal
holdouts stay untouched. Full model rank/dial/latency/HDR/spatial gates remain.

## Duplicate-check clarification, before validator edits / fitting

The prepared source/pixel-unique roles still contain 342 / 221 / 201 repeated
`(f0,label)` keys, but only 4 / 2 / 5 repeated complete 372-feature vectors and
zero conflicting feature-vector labels. A single feature plus a binary label
is not a record identity. Do not discard hundreds of distinct corruption
examples or relax the duplicate percentage to make that surrogate pass.

Extend the existing Parquet validator's C10 with an explicit contract-only
`duplicate_key_columns` option. For these views use origin, width, height and
both decoded-pixel hashes; check every row, require nonnull keys and **zero**
duplicates. Keep the original `(f0,target)` diagnostic and unchanged default
behavior for historical tables. The key columns come from the verified Rust
audit, not filenames inferred after fitting. Preserve raw-table C10 failures.

## Remaining HDR reference screen, registered before fingerprint results

Use the existing audit owner with a private `--native-linear` mode for exactly
12 training PNG sources against the 30 original UPIQ EXRs. Decode with zenpng
and the user-selected zenextras/zenexr, pinned to `109a9ec36727`. Both sides use
the same new fingerprint era: BT.709 linear luminance, nonnegative Y normalized
by its image maximum, `ln(1 + 255*Y/maxY)`, zenresize Lanczos f32 downsampling to
9×8 and the existing horizontal comparison bit order. The maximum only removes
exposure units; no absolute-nit interpretation or display-quality claim occurs.
All-black images yield the zero fingerprint. Reject nonfinite values, alpha,
unknown primaries/transfer and unsupported layouts explicitly. The 30 saved
headers were inspected before this choice: all omit chromaticities, so the EXR
BT.709 default applies. No fingerprint/nearest-neighbor result informed it.

Keep the earlier PNG hash era and results unchanged. Require complete source
hashes/counts and fresh outputs, retain all pairs <=16, and review every strict
<=10 flag and looser <=16 match contextually. Check exact-image and exposure
invariance, unrelated content and malformed/coverage failures before admission.
This remains a crop-blind review aid; also inspect source-family provenance.
The screen does not read human scores or distorted holdout images, change the
training split or expand HDR model-development work. Its sole purpose is to
finish the already-required protected-reference coverage for the prepared fit.

Input-contract clarification before reference fingerprints: the exact-PNG
control correctly refused the existing cleanpicker RGB8 PNG's unspecified
transfer. These source bytes are interpreted as sRGB by the registered native
targeting/extraction recipe. Add an explicit `--assume-untagged-srgb` opt-in for
this caller, recorded per input. It applies only to PNGs lacking ICC, gAMA,
sRGB and cICP metadata, never to EXR or tagged/contradictory color metadata.
The default still refuses unknown transfer; retain that refusal control. This
declares the existing source interpretation rather than silently overriding a
profile or changing the fingerprint threshold after outcomes.
