# Rev5 strip scheduling and retained output

The queue retains the existing 128-row kernels and merges their partials in
producer order. If every active v2 strip/channel job fits the existing 16-slot,
256 MiB limits, one dispatch contains the whole walk. Larger walks retain the
worker-count batch limit. The small-walk choice derives from actual scale and
channel occupancy, rather than a fitted image-size threshold.

Retaining walks use the same queue. Each worker receives disjoint mutable core
rows for the source, distorted, moment and activity destinations. It copies its
own outputs after blur and before feature extraction. Coarse refinement keeps
the original kernel's dense and gradient partials, before accumulation into
running totals. Basic 32-row bands and sparse blockiness still merge in their
original order.

The nonqueued retaining fan-out likewise runs A, retention and B within a worker.
Non-retaining calls keep the original two-iterator channel fan-out; only the
retaining route carries mutable output views.
Self-blur stays disabled when retaining, because retention needs the vertical
planes. Append routes with cross-channel dependencies keep their split fan-out.

Cached source rows are borrowed directly from the caller's immutable pyramid.
The producer advances source-window metadata without copying those rows into a
rolling buffer. Interior queued jobs also borrow cached source windows, avoiding a second
copy. Reflection at image boundaries still uses the existing gather.
Final retained planes remain owned and reusable; workers still copy their core
outputs. This preserves snapshot ownership and buffer reuse contracts.

SAT construction can parallelize horizontal row prefixes. The vertical pass
then performs the original row-prefix-plus-previous-row addition for every
element. It does not reassociate a sum. The existing 64-row grain is used only
when at least one band per worker is available; small grids and callers that
disable parallelism keep the original builder.

feature_v2_kernels.rs, feature_v2_tests.rs and feature_v2_oracle.rs contain
the former inline items, included in the same parent module and lexical order.
The split makes the production walk smaller without changing test assertions.

Validation uses the existing PERF2/SPEEDQ owners via
scripts/rev4_featpot/perf3.py: original native-cache rows, complete map/query
and refinement audits, frozen scorer/refinement/bin gates, then paired
32-clean-round score/map timings. Artifacts and source hashes live outside git;
the final benchmark pointer records their location and checksum.
