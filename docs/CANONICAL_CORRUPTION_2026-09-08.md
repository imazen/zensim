# Canonical corruption model inputs — September 8, 2026 registration

Before new generator edits/data/fits. The September 6 nonlinear HGB study and
Rust ZCTH parity record supersede the September 5 linear-head separability
hypothesis and proposed dial-range guard. HGB achieved 98.52% detection at a
matched 0.5% ladder FP on its historical source split; imposing `dial < 90`
discarded much of the gain. These data came from the superseded inspo tree and
foreign JPEG anchors. They demonstrate a model-form improvement, not current
canonical-family qualification. Preserve them; do not refit the linear guard.

## Existing owners and exact changes

Reuse codec-corpus's `corruption-corpus` generator, now synchronized to upstream
`8e10d4d765667c1c49d74413878fc4bfb46dcf8d` (the older local branch `3e7a8a22`
omitted later real-bug families). Keep catalog/family/seed arithmetic unchanged.
Use its pure RGB generator without its historical `driver` feature. The existing
`zensim-bench/examples/m3_fixture_gen.rs` already owns native PNG IO and the
current zenjpeg anchor encoder; extend it with a private `corruption` mode
instead of introducing another codec adapter. This dependency decision replaces
the preliminary external-driver proposal BEFORE any implementation or data.
No external corpus-library API or feature changes. Pin the pure generator Git
revision, add it to the existing internal `m3-fixtures` feature, and retain the
foreign legacy driver only in its existing historical corpus repository.

The native fixture mode uses current zenjpeg commit
`3d4ad0d77ecb55e78e67828538abb87068180d96`, YCbCr 4:2:0 q10/q20,
independently decoded by the same crate. Require opaque RGB8/RGBA8 PNG sources;
reject transparency/high bit depth instead of silently normalizing them. No
resize. Cache two anchors per source, preserve all catalog seed/params and
inert-case observations, record bitstream/pixel hashes and work counts, and
emit COMPLETE only after every expected output succeeds. The old `resize` and
`jpeg` modes must preserve their emitted bytes. No public API or new feature.
The selected renditions are normalized sRGB. The private mode refuses ICC,
CICP, EXIF, chromaticity and HDR metadata requiring additional interpretation;
it accepts untagged sRGB bytes or an sRGB tag (compatible gAMA only). Every
written PNG is decoded by zenpng and checked against its source packed RGB8
pixels before its file and pixel hashes are recorded.
The final engineering revision also refuses animation/multiple-image inputs;
the measured source corpus consists entirely of static PNGs.

Extend the existing `build_corruption_corpus.py` owner for a strict canonical
manifest mode: exact source bytes/families/splits, no resizing or implicit RGB
conversion, complete keyed rows, explicit producer/feature contract and fresh
output. Keep historical generation explicitly separate and immutable. Use the
existing Rust feature extractor, `train_corruption_head.py` estimator/exporter,
ZCTH reader, `BakeScorer` composition and evaluation owners. Do not introduce a
second corruption classifier, scorer or generator.
Pin extraction to formula revision 1 with the libm root form and explicit raw
IDs 0..371. Record all local package source hashes, resolved Cargo metadata and
lockfile, codec revisions, generator/extractor binary hashes and build options.
Both binaries and producer source files must remain unchanged throughout a run.

## Bounded first dataset and checks

Use the existing canonical 12 train / 8 validation source manifests from the
September 8 native targeting experiment, at their stored longest-side-256
geometry. No terminal family. Generate the current full catalog, preserving
inert/saturated cases with measured pixel differences.
Inert entries retain their catalog identity and params but receive
`is_corruption = false`: an unchanged image cannot truthfully be a positive
corruption label. Duplicate pixel renditions are retained for catalog audit and
must be collapsed or weighted once per origin/pixel identity before fitting.
q10/q20 anchors are
actual native encodes, not assumptions about which score they reach. Supplement
honest negatives from the retained native JXL/AVIF train bounds and separately
from validation bounds, after exact source-family and byte checks. Do not fit
on the validation family records. A later fit registration must specify source
partitions for probability calibration separately from held-out evaluation.

Before a fit, prove complete generation/extraction and current raw feature
parity, hashes, split/near-duplicate compliance and label identity. Exercise
odd dimensions and a >256 multi-group geometry in engineering controls.
Keep a golden hash of every old pure corruption output to show only the IO/
anchor paths changed; JPEG bytes must equal the explicitly configured native
encoder and decode back to the saved anchor pixels. Failures must return
nonzero and cannot leave a COMPLETE marker. Validate the complete Rust-served
composition, including its unchanged dial scores on honest controls. This
packet does not waive spatial discontinuity/unsupported-term handling, G-ADDR,
independent matched-RD, realistic targeting or final model qualification.

## Result and replay

The [completed data packet](../benchmarks/canonical_corruption_2026-09-08.md)
records exact coverage, retained duplicates, controls and remaining admission
work. This registration does not authorize calling the raw catalog a
deduplicated training view or calling a new model qualified.

```sh
# Run inside zensim-bench; use the recorded Cargo.lock for exact replay.
cargo build --release --example m3_fixture_gen --example extract_features_372col \
  --features training,zen-decode,m3-fixtures

# From zensim root. All paths explicit; DATA must be a fresh output directory.
python3 scripts/v_next/build_corruption_corpus.py \
  --sources-json SOURCES.json --family-manifest split_map_family.tsv \
  --producer-json PRODUCER.json --artifacts-dir DATA --out corpus.parquet \
  --gen /path/to/m3_fixture_gen --extract /path/to/extract_features_372col
```

The canonical JSON mode fixes the full source list and 372-column revision-1
contract, retains PNGs and stops on any error. The existing `--sources` TSV
mode remains the historical PIL/image-anchor replay path described in the
script's historical header; it must not create a canonical data packet.
