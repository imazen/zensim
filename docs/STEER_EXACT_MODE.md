# Exact block gains for the STEER research experiment

`ZENSIM_STEER_EXACT=all-blocks-v1` is an explicit research mode, selected when
`BakeScorer::prepare_steering` creates a session. Absent or `off` preserves the
existing approximate map. Unknown values refuse. This is diagnostic steering,
not production adoption or a change to legacy linear M2.

The mode requires an isolated Rev5 SDR sRGB RGB8 bake and bin1. On every
`SteeringSession::compute`, it eagerly pastes the reference into each rectangle
on the8/16/32/64 grids, independently from the original reconstruction, then
runs full native extraction and the complete score. `refinement_gain` returns
that finite gain for registered rectangles. Unsupported rectangles return NaN;
no approximate fallback or subset-delta forward is substituted. Ordinary
comparison score, features and explanatory density retain their existing paths.
Input rows may be strided; padding is never treated as pixels. Codec routing,
HDR, other pixel formats, ensembles and score dispositions are refused.

All repair extraction, forwards, allocation and gain-table construction occur
inside the cached map call. Include them in the unchanged spatial/RSS collector.
No performance or qualification claim follows from exactness. The complete
native reference deliberately favors an auditable repair over unmeasured cost.

`just steer-exact-tests` checks91 synthetic edge-clipped rectangles against
independently constructed native repair forwards, including strided input,
identity and unsupported queries. These deterministic generated images are
outside the fixed135-case steering panel. They read no corpus, model or labels.

`steer_exact_smoke` (the unpublished validation tool) accepts a bake plus its
expected SHA256, generates two fixed synthetic geometries and checks every
8/16/32/64 repair against independent complete native forwards. It records
ordinary/preparation/exact-map/oracle wall components and pixel/model hashes.
It never opens a panel or label table and does not evaluate steering quality.
`steer_exact_smoke.py` supplies its manifest-bound non-installable FitCell
wrapper. External GNU time supplies process peak RSS; build timings are not
serving measurements. The smoke policy is separate from registered fit caps.

`--compare-bake PATH --compare-sha256 SHA` additionally compares native ordinary and repaired-score bits against a hash-bound matched control on the same procedural pixels. The receipt exposes only parity flags/counts and hashes; it is an engineering replay check, with no targets or quality statistic.

The scientific launch owner `e33_launch.py` accepts the versioned STEER packet and fixed eight-jobset inventory. Its first-cell mode refuses expansion until a hash-bound native-image persistence receipt records a drained worker. `harvest` and `harvest-cycle` enter the pinned image before importing the archive verifier or reading ledgers. A failed jobset remains explicit while the cycle attempts the others. Preparation alone does not authorize queueing.

STEER harvest proves a never-populated remote ledger empty with an independent nontruncated listing before attempting sync. Empty remote state with cached local ledger evidence refuses, and nonempty-prefix sync/authentication failures keep the existing refusal behavior. Its image wrapper creates the disk-backed TMPDIR child before invoking store tools. The fit-image builder exercises AWS CLI, s5cmd and Python imports; copying a standalone AWS launcher loses its installation-relative shared-library lookup, so the image preserves the installation symlink.
