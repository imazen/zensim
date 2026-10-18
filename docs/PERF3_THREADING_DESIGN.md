# Rev5 strip scheduling and retained output

The queue retains the existing 128-row kernels and merges their partials in
producer order. If every active v2 strip/channel job fits the existing 16-slot,
256 MiB limits, one dispatch contains the whole walk. Larger walks retain the
worker-count batch limit. The small-walk choice derives from actual scale and
channel occupancy, rather than a fitted image-size threshold.

Retaining walks use the same queue. Each worker receives disjoint mutable
core rows for the source, distorted, moment and activity destinations. Canon64
Rev5 v2 cells write the five moment/activity cores directly into their final
buffers, using row windows of the original local blur pair tree. Mean-reference
halo rows stay in scratch for the activity chain. Basic 32-row bands read the
same vertical core values with radius zero and merge in their original order.
Coarse refinement keeps the exact dense and gradient kernel partials, before
accumulation into running totals.

Direct moment output serves queued walks and single-worker fused walks. Small
parallel walks outside the queue, other arithmetic modes and other feature
families retain the existing copy path. Non-retaining calls keep the original
two-iterator channel fan-out; only retaining workers carry mutable output views.
Append routes with cross-channel dependencies retain their split fan-out.

Cached source rows and interior queued source windows are borrowed directly from
the caller's immutable pyramid. Reflection at image boundaries still uses the
existing gather. Source and distorted core planes remain owned and copied for
snapshot buffer reuse; the five moment/activity retention copies are removed
on the direct path. The original full-plane blur kernels remain unchanged.

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

The ordered central-moment block has one shared out-of-line body. This keeps
its small scalar loops out of the dense and gradient SIMD entry bodies without
changing a term, reduction or merge order.

The additional direct-core work is reviewed with the cached scoring API rather
than advanced on main: its qualified run retains a significant score cost at
1024 squared with eight workers. The benchmark pointer records that trade-off
alongside the map gains and unchanged output-bit audits.

The final combined API binary has no significant score regression across its
nine paired cells. The isolated-parent cost above remains a separately scoped
measurement. Full first-32-clean percentiles and both source identities are in
[the PERF3 report](../benchmarks/perf3_2026-10-18.md).
