# Retain paired measurements for real latency assessment

September 14, 2026. Measurement infrastructure; no candidate qualification.
The existing Zensim speed benchmark previously saved only aggregate summaries.
Those cannot establish the scorecard's p95 requirements. The canonical zenbench
owner now retains completed measurement rounds in its existing result JSON.

The benchmark pins reviewed zenbench revision
`16fb8fddffdb90d944aec00d96a8358df6507358`. It preserves both raw and
overhead-compensated total nanoseconds, actual per-round iteration counts,
and randomized execution order. Timing vectors use benchmark declaration order;
rounds remain chronological. Existing summaries are computed as before.
Warmup and resource-gate waits are excluded. Retention itself does not certify
quiet conditions, representative inputs or any model quality.

Use `ZEN_S2_SINGLE_CALL=1` and `ZENBENCH_RESULT_PATH=<new-file>` through the
existing `zensim-bench/benches/ssim2_speed_bar.rs` owner. For individual-call
latency distributions, verify that **every actual round has one iteration**;
use the raw total durations. Dividing a batch time does not recover individual
call tails. Preserve hardware/build/input/model identities and contention
records, and enforce the required number of accepted paired rounds separately.
Historical summary-only files remain insufficient; no p95 is inferred from them.

Multi-run aggregation clears the retained samples under every policy, including
Best: different arms can select different winning runs, so they do not form one
paired population. Keep the original single-run artifacts. Historical JSON and
immediate-mode compatibility results deserialize with empty samples.

## Verification

End-to-end owner tests cover paired round order, variable batches, single-call
records, raw/compensated consistency, summary reconstruction, serialization,
historical compatibility, and clearing on aggregation. Default and minimal
feature builds pass. All 142 active library tests pass (one ignored); Clippy
passes the repository's default, WASM and criterion-compatibility configurations.
Public API snapshots regenerate and check successfully, also recording the
already exported resource-gate API missing from the prior snapshot.

Initial Clippy caught stale test-fixture constructors; all three affected
fixtures now initialize an empty samples field. The first snapshot build exhausted disk space;
removing only the reproducible Zensim debug incremental cache recovered 78 GiB.
Source, model and experiment evidence were preserved. Original failed logs remain
in `~/work/zensim-validation-2026-09-14/latency-evidence/`.

The actual release-built Zensim benchmark smoke passes: two synthetic geometries
(64x64 and 128x128), fast-ssim2 and the complete named D surface, three rounds
each, exactly one call per observation. All 12 observations retain raw and
compensated timings with pairing/order metadata; reconstructing the aggregate
means matches the saved summaries. These tiny synthetic fixtures and three
rounds do not meet the performance qualification contract.

[Structured checks](latency_evidence_2026-09-14.results.json) and the
[served packet](http://localhost:3300/zensim/reports/latency-evidence-2026-09-14/latency_evidence_2026-09-14.md)
retain the registration, original failures, checked output, commands and hashes.
This change supplies auditable observations for future complete performance
campaigns; it does not supply missing scalar/map p95, worker RSS, HDR coverage,
quality gates or native spatial rate-distortion evidence.
