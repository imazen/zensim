# Corruption-head arithmetic contract — September 14 registration

Before implementation or model fitting. The existing Rust tree-head attachment
checks available slots but does not compare feature arithmetic with the base.
The strict trainer currently produces D228/Rev1 heads only. Reusing those
weights on a Rev3 walk is not a valid new model, even when the IDs match.

Concrete callers are `BakeScorer::with_corruption_head`, its scalar and prepared
paths, and `CorruptionHead::check_servable_by`. Preserve their signatures. Add
private revision state and checks; no new public API item. Raw feature-vector
callers still own the provenance of the vectors they supply.

First demonstrate the legacy-head/Rev3-base attachment failure with a synthetic
stump and an explicitly stamped base. Then require matching revisions in both
planner and non-planner builds, and in named-profile admission. Legacy ZCTH
v1/v2 mean native-pyramid Rev1, preserving their bytes and numerical inference.

For new heads, extend the existing format/exporter to ZCTH v3: f32 inputs,
formula revision 1/2/3 stored as u32 at header offset52 and included in the
shape hash. The section layout stays unchanged; metadata remains provenance.
Old readers refuse the new version. New-format native-pyramid heads remain
incompatible with fractional sampling, which requires a separate contract.
No feature-plan widening is introduced. The default exporter remains byte
compatible with v1/v2; explicit revision stamping opts into v3.

Verify wrong-revision refusal, valid attachment for each supported revision,
unknown/mutated revision rejection, old precision behavior, Python-export/Rust
parity, and actual scalar/prepared composition on synthetic pixels. Run the
relevant feature builds, public API checks, lint and Clippy. No scientific
dataset is needed for these engineering checks. This enables a correctly bound
Rev3 training experiment; it does not train or qualify a corruption model.

## Reproduction and implementation

The pre-fix test executed under Rev3 and failed because a Rev1 stump reading
f13 attached successfully to a base explicitly declaring Rev3. The failure was
in admission, before pixel scoring; the log is retained. The corrected test
rejects that composition. Linear companions already checked revisions.

The Rust loader now interprets legacy v1/v2 as Rev1. Format v3 appends the
u32 revision to the existing shape descriptor before FNV hashing, and stores
that value at offset52. Unsupported values and valid-value mutations without
a matching hash are refused. Nonzero legacy reserved fields are refused.
The JSON provenance block does not override the wire contract. Fractional
sampling and requests for uncomputed features remain refused.

The existing Python exporter accepts explicit `formula_revision=1|2|3` only
with `input_precision="f32"`; that opt-in selects v3. Omitting the revision
preserves existing v1/v2 bytes. No scientific head or default is replaced.
The strict training manifest remains the previously registered Rev1 recipe;
a new Rev3 fit still needs separate data/recipe admission.

## Verification

- 27 Rust corruption-head tests pass, including the pre-fix regression,
  named-profile refusal, precision/hash controls, all three matching revisions,
  and mismatched base/head combinations. Each revision's pixel test runs in
  its own correctly configured process.
- Scalar and prepared composition agree on synthetic 96×96 pairs. An inactive
  head preserves the base score and rectangle map queries. An active head
  returns `CorruptionDetected` even when a negative base hides activation in
  the scalar minimum. Malformed calls do not remove protection.
- The no-planner feature build passes its actual revision-admission test.
  It retains three unrelated existing unused test-helper warnings.
- `verify_corrhead_format.py` fits one synthetic 16-tree fixture and invokes
  the canonical Rust parity tool on 512 rows for each of five format/revision
  cases. All 2,560 raw predictions match at zero ULP, probabilities match
  exactly, and activation sets agree. This is serialization evidence, not
  learned perceptual quality or extraction parity across revisions.
- The pre-change and current exporters produce byte-identical v1/v2 fixtures.
  Three invalid writer contracts are refused before output creation.
- Script lint checks 616 scripts; Clippy and the pinned public API check pass.
  No public signatures or snapshots change. [Structured receipt](corruption_revision_contract_2026-09-14.results.json).

Replay the format check with an explicit exporter saved from pre-change commit
`eb3df151b86040a38eedd0eb258739012f44e0b2` and a freshly built Rust tool:

```sh
cargo build -p zensim-validate --bin corrhead_parity
python3 scripts/verify_corrhead_format.py \
  --baseline-exporter /path/to/pre-change-train_corruption_head.py \
  --parity-bin target/debug/corrhead_parity --out-dir /path/to/fresh-output
```

The first pixel fixture changed one pixel in an otherwise identical image.
Rev2 hit its existing bounded-SSIM assertion (`d=-0.000012874603`), before
reaching a new-head comparison. The failing source/command is retained in
the evidence packet. Composition checks use a whole-image contrast change
instead; no SSIM formula, tolerance or assertion was changed. This does not
establish near-identity numerical correctness for Rev2. A second wrapper
attempt also exposed the existing scalar process/revision requirement;
separate configured processes now exercise all intended cases.

The [served evidence packet](/zensim/reports/corruption-revision-2026-09-14/index.html)
contains the failed and successful checks, synthetic fixtures and replay inputs.
No scientific dataset was opened and no EVAL or TEST was used. A properly
admitted Rev3 corruption fit with broad honest/native controls remains next;
this serving correction does not change any model's qualification verdict.
