# Label-free engineering judges

Private executable tools; no library API. Build this standalone manifest with
`cargo build --release --locked --manifest-path tools/shipgates-judge/Cargo.toml`.
Freeze the zensim core and patched pixel dependency before collecting results;
retain the source commits, lockfile and executable hash with the measurements.
The SHIPGATES fixed judge uses core `9bf855bc` and pixel source `c6ca8557`.

`shipgates_judge --pairs pairs.tsv --out-dir fresh-directory --ssim2-map-policy scalar-only`
reads explicit `ref_path`/`dist_path` columns, preserves every input column, and
scores opaque sRGB8 PNGs with SSIMULACRA2, butteraugli and fixed zensim B.
The scalar-only SSIMULACRA2 map exception requires explicit caller authorization.
Both available pixel maps are retained losslessly by hash in PNG byte containers.

`shipgates_prepare_source --crop-opaque input.png fresh-output.png` selects the
largest all-opaque rectangle and preserves its RGB bytes, including strided rows.
Its JSON receipt records the crop, source hashes and executable hash. The earlier
white-matte prototype differs from an exact IEC sRGB oracle at some 8-bit
rounding boundaries. Its failed diagnostic is retained; current preparation
refuses non-crop invocations.


`shipgates_prepare_source --expand-png8 input.png fresh-output.png` performs
lossless palette/grayscale expansion with every source pixel opaque. It emits
8-bit RGB or Gray PNG using existing zenpng developer/statistics and gray8
encoders, then checks every delivered RGB byte. It rejects sixteen-bit inputs
and color metadata outside the declared sRGB contract. The crop mode keeps its
original input policy; matte conversion remains refused.

The extra zenpng developer feature serves source preparation only. The campaign's
archived fixed-judge executable and its original manifest remain frozen; adding
source-container support does not rebuild or replace that executable.
