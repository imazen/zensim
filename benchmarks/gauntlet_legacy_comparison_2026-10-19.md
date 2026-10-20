# Matched comparison with the existing B/D profiles

Here **B** denotes the historical linear bake `a96a5a66…`. After this
comparison, the owner assigned diff+frag-820 (`89cb4861…`) to the public B
slot. The frozen measurements below retain their original identities; see
[the current B record](../docs/PROFILE_B_DIFF_FRAG_820_2026-10-19.md).

C / diff+frag-820 has higher signed SROCC than B on four of five exposed
external panels and D on three. W-C has higher point estimates than B on
all five. These are descriptive comparisons, not significance tests or a
new selection rule. Research C is distinct from the named public Profile C.

| Panel | B | D | diff-410 | C / diff+frag-820 | W-C |
|---|---:|---:|---:|---:|---:|
| CSIQ865 | .934859 | .901775 | .944465 | .953279 | .961758 |
| PIPAL23200 | .567189 | .497781 | .613416 | .607488 | .617801 |
| CID22-B2100 | .889964 | .879456 | .905267 | .909689 | .904991 |
| KonJND JPEG SELECT404 | .570556 | .588287 | .521301 | .511471 | .589398 |
| LIVE267 | .875359 | .957702 | .932257 | .907877 | .889548 |

The three research candidates retain their original full-population results.
Their four D1 panels use whole-dataset-withheld fits, where W-C leads C on
all four. Across those four plus the five external panels, W-C leads C on
seven of nine; C leads diff-410 on five. Comparable B/D withheld fits are
unavailable, so no B/D held-out result is invented for the D1 panels.

## Native execution and limits

Existing B `a96a5a66` and D `cd1098b4` are copied byte-identically from the
current named profiles. Fresh native **Rev1** features score exactly the
same external pixels and labels as the Rev5 candidates. No Rev5 feature
column is substituted into the legacy arithmetic. Historical training and
development differ; the original LIVE known-overlap exclusions remain,
but exhaustive historical crop/resize clearance is not claimed.

All26,836 B predictions agree exactly with the native pixel, literal,
cached and stored-f32 surfaces. D's production feature predictions agree
exactly with18 fixed native pixel audits spanning all five panels.
Consumed-feature audit deltas are zero. The extractor groups CSV output by
reference, so its explicit integer row identity restores original order;
the failed initial positional assumption and its log are retained.

The canonical native panel owns signed SROCC, fitted PLCC, KROCC and the
scatter diagnostics. SROCC intervals resample whole references10,000 times,
seed20261017; LIVE's10-reference intervals are descriptive. The full
gauntlet now puts every measured corpus in its main comparison table,
including PIPAL, CID22-B and JPEG SELECT. Unmeasured product qualification
and composites remain unmeasured. Existing browser gates pass unchanged.

The existing corruption companion is a separate LIVE-only diagnostic:
C .907877 becomes .877399 when attached. Extreme/milder manfishing scores
are49.714/-13.915 without the head and0/-13.915 with it. Both B8.569/8.756
and D-213.149/-84.058 order that pair correctly. No new head was fitted.

Native extraction ran under the shared lock/run-heavy: rc0,241s,
peak-RSS0.33GiB, min-available40,769MiB, peak-load1.53. Assessment/parity/
canonical statistics: rc0,85s, peak-RSS0.90GiB, min-available40,138MiB,
peak-load1.30. These are execution receipts, not speed benchmarks.

[Artifact locations and provenance](gauntlet_legacy_comparison_2026-10-19.pointer.md).
No profiles, thresholds, gates or selection decisions changed; protected
populations remain untouched.
