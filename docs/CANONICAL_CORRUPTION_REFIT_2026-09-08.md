# Canonical corruption refit — September 8 registration

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
