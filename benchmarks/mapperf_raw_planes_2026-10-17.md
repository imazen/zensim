# Reuse raw distorted planes for prepared finite refinement

Prepared Rev5 subset maps omit full-resolution X/B feature slots. Their finite
queries still need the raw X/B pixels. The producer previously stored only Y at
scale zero, so snapshot construction converted the distorted image to XYB again.
After production-strip reuse, the 1024² C snapshot still measured 7.194096 ms
in the instrumented release probe.

Prepared retaining walks now keep the raw distorted scale-zero planes while
continuing to gate every feature with the original computation plan. Capture
moves these planes and skips its second distorted-image conversion. Ordinary
and scalar subset walks keep their prior storage path. Reused buffers are sized
separately from the phase-A planes, which remain absent for unserved channels.
Declared/computed raw counts remain 410/410 for diff-410 and 420/420 for C and W-C.

The independent oracle checks every retained pixel bit against reconversion,
every feature and mean-offset bit against ordinary sparse extraction, finite
repair bits, odd and multistrip geometry, and prepared-to-ordinary buffer reuse.
The scorer and refinement suites pass, including this oracle. Fresh C and
diff-410 before/after audits pass all four required sizes, including every
score, declared feature, gradient, density, rectangle query and finite probe.
The owner-approved ten unread diff-410 fields alone become structural zeros.

C's matched CPU10/v4x/one-thread results for the required large sizes:

| Size | Original before p95 ms | After p95 ms | Butteraugli p95 ms | After−butter 95% CI ms |
|---|---:|---:|---:|---|
| 1024² | 74.384539 | 58.228047 | 63.134925 | [-16.530792, -3.809775] |
| 2048² | 306.723019 | 248.473391 | 352.239878 | [-106.339624, -93.690192] |

These are exploratory engineering measurements on one frozen pair per size,
using the unchanged SPEEDQ quiet admission, first32clean-of64, and existing
Rust p95 owner (10000paired resamples,seed20261016,type7). The remaining model
and size cells, four-size fits and complete native-cache replay are still in
progress. They are not implied by the two completed C rows.

Source pins, binaries and raw evidence are outside git through the
[artifact pointer](mapperf_2026-10-17.pointer.md). No public API or numerical
tolerance changes.
