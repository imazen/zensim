# MAPPERF declared-feature audit (2026-10-17)

The three Rev5 candidates use the production structural-read planner for both
scalar scoring and complete prepared maps. They do not invoke the comprehensive
research extractor.

Frozen-bake audit before the declared-subset fix:

| Bake SHA prefix | Model inputs | Declared/live raw slots | Computed raw slots |
|---|---:|---:|---:|
| 7cf4cfd9 (diff-410) | 410 | 410 | 420 |
| 89cb4861 (diff+frag-820) | 820 | 420 | 420 |
| f73059ff (W-C) | 820 | 420 | 420 |

The 820-input models combine direct values and derived products. Their extraction
request has 420 distinct raw dependencies. All three plans emit into a
720-position identity row, with structural zeros outside their populated slots;
the row width is not a compute count. Full-resolution X/B, peaks, masked/IW,
append, append2 and later banks are disabled.

The diff-410 extras are the ten active-cell reference-only PJND_FRAGILITY slots:
422, 480, 509, 538, 567, 596, 625, 654, 683 and 712. Their finalizer uses source
gradient sums already required by declared difference features. Omitting this
finalizer cannot eliminate the gradient kernel.

Source owners: `metric/bake.rs::plan` → `feature_plan::Plan::for_bake` →
`feature_plan::bake_read_slots`, including derived-input dependency expansion.
Scalar Rev5 scoring passes the plan to `fold_engine::compute_fold_backed` and
`feature_v2::compute_folded_v1_372_streaming_impl`. Prepared maps pass it to
`Fused944Session::planned_features` and
`feature_v2::compute_folded_with_ref_retained`, then evaluate density from retained
state and capture a finite-refinement snapshot.

The native diagnostic loads each pinned bake, resolves the actual serving plan,
and records declared IDs, live reads and `ComputeSet::populated_slots`. Its
explicit test entry is `metric::bake::mapperf_audit::declared_plan_and_stages`
under `--cfg steerfix_instrument`; callers select `MAPPERF_AUDIT_ROOT` and
`MAPPERF_AUDIT_MODEL`. It never discovers external data implicitly. Raw receipts and qualified test-profile timing are retained in the
[artifact pointer](mapperf_2026-10-17.pointer.md). Test-profile timings are not
matched release-latency evidence.

The approved correction skips fragility finalization when its raw IDs are unread,
including finite-refinement replay. Native post-fix counts are 410/410 for
diff-410 and 420/420 for both derived-input bakes. A forced-original-plan oracle
passes at 64², 256², 1024² and 2048²: every score, sensitivity, density value,
grid rectangle query and finite-refinement probe matches bitwise. Only the ten
undeclared reference-only output positions become structural zeros.

The owner directed a fresh complete matched run on quiet CPU10, excluding CPU2
and its SMT sibling CPU14. CPU10 and sibling CPU22 were checked with per-core
load samples. Every before/after/butteraugli arm is pinned to CPU10. The unchanged
measurement owner records the chosen CPU in each header. Instrumented release
stage medians at 1024² (milliseconds, ten repeats after three warmups):

| Candidate | Scalar extraction | Model | Map extraction | Map forward | Gradient | Density | Snapshot |
|---|---:|---:|---:|---:|---:|---:|---:|
| diff-410 | 27.218621 | 0.008667 | 40.660831 | 0.017784 | 0.605294 | 19.789433 | 13.418700 |
| diff+frag-820 | 27.285257 | 0.016331 | 33.063214 | 0.027968 | 1.192434 | 19.767566 | 13.422238 |
| W-C | 27.443969 | 0.006563 | 41.026725 | 0.012674 | 0.342781 | 19.983041 | 13.623670 |

Scalar extraction uses the uncached-pair production surface; complete maps use
a prepared reference. These instrumented diagnostics are separate from the
paired p95 acceptance run, which remains pending. Stage medians need not add to
the median total. run-heavy: rc=0, 4 s, peak RSS 0.17 GiB, minimum available RAM
22843 MiB, peak load 1.92.
