# Native color admission — September 14, 2026

Registered before implementation and new pixel reads, after the
[color/HBD/HDR status review](color_hbd_hdr_status_2026-09-14.md).

Concrete caller: the existing `verify_bitstream_decode` tool needs an explicit,
hash-bound inspection of the 33 P3 TRAIN references and their 180 stored codec
reconstructions, plus the previously unresolved AVIF q85 control. Source-file
metadata must be distinguished from decoded-pixel semantics before defining a
corrected extraction contract. XYB conversion makes treating every source ICC
as the decoded RGB profile unsafe.

Extend `shared/zen_decode.rs`, the existing decoder owner, to retain native
`PixelBuffer` and original codec metadata. The existing RGB8 function becomes
an explicit projection of that same result, with unchanged conversion arithmetic.
Do not add a second decoder, color converter, metadata parser, feature extractor,
or scorer. No library public API or named profile changes.

Add `verify_bitstream_decode --inspect-list LIST --out FRESH.jsonl` under the
existing `zen-decode` feature. LIST is a TSV with explicit `path` and `sha256`;
reject malformed/duplicate entries, hash mismatches and existing output before
scoring or publication. Inspect source color fields, decoded descriptor/context,
native active-row bytes and legacy RGB8 projection. Preserve source ICC hashes,
known-profile identification and its valid-use limitation; never guess a color
transform from a profile description. This mode performs no metric scoring.

Validate with synthetic imazen encodes: exact native 16-bit low-bit retention,
PNG ICC/CICP retention, JPEG/AVIF/JXL/WebP source-versus-output metadata, and
unchanged legacy RGB8 decoding. Native hashes exclude stride padding and bind
descriptor, dimensions and stride separately. Refusals must leave no successful
partial report. All real inputs must inherit the existing product TRAIN
admission and family roles; no EVAL/TEST/TERMINAL inputs or labels are admitted.

Then inspect the exact admitted source/codec outputs and report what is known
about profile retention and color semantics. Decoder correctness and metadata
retention do not qualify a model or prove the historical encoder handled color
correctly. A corrected color/precision extraction era and its public-score parity
remain necessary before restoring the affected training coverage.

## Completed native inspection

215/215 files decode and reproduce their prior RGB8 pixel hashes exactly after
the decoder refactor and codec synchronization. No metric score is produced.
The list covers 214 existing TRAIN-fit pairs: 213 affected P3 pairs (including
33 identities) and the one previously unresolved AVIF q85 pair. File-hash,
family and prior-admission checks precede pixel access. The first admission
attempt incorrectly expected the color audit to reference the product manifest
directly; it actually references the honest-control admission. That attempt
refused before pixels; the corrected join verifies both immutable authorities.

| Source format | Files | Native source-depth coverage | Color finding |
|---|---:|---|---|
| PNG | 34 | 8 bit | 33 references carry the same 520-byte P3 ICC; one untagged q85 reference. The decoder exposes these ICC bytes in source metadata but its pixel context is absent and descriptor transfer is Unknown. |
| AVIF | 43 | 22 eight-bit, 21 ten-bit | All declare BT.709/sRGB/full range; 29 use matrix 6 and 14 RGB identity matrix 0. Ten-bit outputs remain Rgb16 in the new native result. |
| JPEG | 42 | 8 bit | 36 have no ICC. Six retain a 720-byte ICC and Unknown decoded primaries/transfer; their original configurations explicitly name XYB. Do not reapply that source profile to already transformed samples. |
| JXL | 54 | 8 bit | All report authoritative sRGB CICP and a generated 536-byte ICC. Decoded descriptor is sRGB; pixel context is absent. |
| WebP | 42 | 8 bit | No ICC; decoded descriptor is sRGB. |

**None of the 180 P3-derived stored reconstructions carries the source's
520-byte ICC.** That alone does not prove a color bug: a correct encoder may
convert to sRGB before encoding. It also does not authorize interpreting all
stored reconstructions as P3. Original encoder input/conversion provenance must
settle that question. The old decoder's raw P3 reference hashes establish its
own missing interpretation, not what the encoder did.

The existing native `zenpixels::icc::identify_common` does **not** recognize
the 520-byte profile. It also returns no match for the six XYB JPEG profiles
or generated JXL profiles. Identification is therefore recorded as null rather
than invented from a description. Use the existing full CMS route when profile
authority actually requires it; authoritative JXL CICP and codec-owned XYB
transforms must remain distinct cases. No new ICC parser or profile hash table
was added.

The q85 case retains exactly the earlier distorted-pixel hash and reports
BT.709/sRGB/full-range matrix 6 through the native decoder. This independently
supplies native metadata evidence, but does not locate the cause of its cast or
change its unresolved label.

## Implementation, checks and provenance

`decode_native_bytes` retains the codec's typed PixelBuffer and complete
PngInfo/ImageInfo. Source metadata and decoded descriptor/context are separate.
The unchanged `decode_rgb8_bytes` surface projects that same output through
the existing RowConverter/flattening owner. There is no new color transform,
decoder fallback, dependency, named profile or library API. Native precision is
now accessible; the existing feature extractor still requests legacy RGB8.

17 self-contained codec tests pass, including exact native u16 samples with
varying low bytes, retained P3 ICC, retained PQ CICP and XYB metadata. The
inspection-list unit test and five real CLI refusal checks pass. The latter
cover duplicate/invalid rows, missing files, a valid prefix followed by a bad
hash, and existing-output refusal before input access. No rejected case emits
a successful partial report. Targeted bench clippy passes; dependency warnings
remain visible in logs. This is engineering and input-provenance evidence,
not HBD/HDR model qualification.

Dependency synchronization found new upstream color/container fixes: zenavif
advanced from `a26343da` to `a7c56be9`, and zenjxl-decoder from `8e3edef` to
`940d2c5`, by clean fast-forward. Native tests were rerun afterward. Other
consumed sibling tips were already synchronized; unrelated JPEG submodule
files were preserved. New AVIF commits include payload/container agreement,
range and metadata corrections, but do not prove the cause of an older file.

The compact `native_color_admission_2026-09-14.results.json` pins the binary,
source/lockfile identities, source revisions, admission, all 215 records and
test logs. The served report includes the exact input manifest and inspection
records. Original files and all previous experiments remain unchanged.

Next: define the color/precision-aware scorer contract using existing CMS and
public ImageSource owners, and trace the original encoder input interpretation.
Do not restore these rows as clean training negatives merely because metadata
is now retained. Color-correct feature/scalar/map parity and representative
model qualification remain open.
