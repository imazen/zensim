# Reuse production strip partials in refinement snapshots

The complete Rev5 prepared-map path retained its coarse planes, then ran the
dense and gradient kernels over those planes again to initialize refinement
snapshots. The instrumented 1024² release probe measured snapshot medians of
13.418700, 13.422238 and 13.623670 ms for diff-410, C and W-C respectively;
these stage diagnostics are separate from paired latency acceptance.

Retaining walks now preserve the original DenseAccum and GradientAccum values
for each served strip at scales 1–3, in production order. Copying capture copies
these partials; moving capture moves them. Snapshot construction checks the
expected strip count instead of rerunning unchanged kernels. Repair queries
still recompute touched strips and merge in the original order. Scalar walks
do not retain partials; older revisions retain their existing refinement path.
The retention resets partial storage on every walk, including moved-buffer reuse.

The independent oracle reconstructs the old capture pass and compares every
finite-feature delta bit for reference and explicit planar candidates across
multiple strips, odd dimensions, changed dimensions and moved-buffer reuse.
The native scorer/refinement suites pass. The corrected normal release binary
also passes the full before/after C audit at64²/256²/1024²/2048²: declared features,
score, sensitivities, density, rectangle queries and finite probes are bit-exact.
Final matched latency and complete native-cache replay are still in progress.

Run the native suite with `just mapperf-frozen-map-check <binary>`; the diagnostic
build is `just mapperf-audit-build` under the shared heavy-work guards. Normal
release timing uses the locked addrfix-runtime and the existing SPEEDQ owner,
with frozen before/after/butteraugli arms pinned to CPU10. See the
[artifact pointer](mapperf_2026-10-17.pointer.md) for preserved source pins,
receipts and data. No public API or numerical tolerance changes.
