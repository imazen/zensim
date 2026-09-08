# Cleanup completion — September 7 local / September 8 UTC, 2026

The operative September 7 [authorized list](../docs/PLAN_CRUFT_PURGE_2026-09-06.md)
is complete. Conditional removals were resolved using the stated evidence
criteria: useful shared features, buffered compatibility paths and unproven
loss/head removals remain. The work does not qualify a new common-dial model.

Start development at [SESSION-RESUME](../SESSION-RESUME.md), then
[WAVE_PLAYBOOK](../docs/WAVE_PLAYBOOK.md). The end user controls one target score;
model rank, codec floors, identity, target error and total loop cost are separate
measurements. B is the default and D the explicit fast baseline; B/C/D are
replaceable. Python supports invention/oracles; every introduced model and its
full composition must serve and evaluate through the Rust surface.

## Completed changes

- Added the reusable `zensim::BakeScorer` candidate API and migrated candidate
  evaluation to it, including heads, pins, splines, codec corrections, ensembles
  and corruption composition. Removed duplicate runtime/metadata dispatch.
- Consolidated transform screening, spline writing, feature/header admission,
  explicit IDs and training capability checks. Removed positional CLI training
  and silent truncation. Historical replay is explicit and cannot qualify.
- Corrected C/CHdr training-versus-serving activity semantics; converted both
  to explicit IDs and retired two alternate planners. The score correction is
  intentional and documented, with consumed-feature, HDR and feature-build gates.
- Made evaluation stages reusable by complete content identity and atomic stage
  completion. Qualification checks every member/companion's training and table
  provenance; unknown decoder sidecars invalidate reuse and cannot confer a pass.
- Retired **25 obsolete source files** across the approved batches, including
  the old training/baking/calibration families, screen adapter, preview/example
  and launch tools. Kept exact sources in Git/private archives and unique live
  algorithms, data, bakes, tests and numeric references.
- Reduced CLAUDE to current instructions, made the playbook the workflow owner,
  archived historical root plans/notes, synchronized current profile docs, and
  resolved stale TID split policy and all 26 old hygiene findings without exemptions.
- Extended the existing target controller and demo to measure complete Rust
  candidates, emitted bytes and independently judged reconstructions. The default
  CLI follows `codec-target`; JPEG/PNG measurements use the Imazen decoders.
- Updated both gauntlet pages: shared score axes include negative and above-100
  tails where present; the default set is B/D and the three constrained controls;
  qualification precedes composite. Added the three rav1e results and refreshed
  H seed 4004 through the promotion owner, preserving its six scientific panels
  exactly. Full source verdicts remain on disk. The page omits undisplayed,
  duplicated reproduction metadata while retaining every displayed input/argv
  field and coverage record.

## Scientific acceptance

[Nine real training fits](cleanup_scientific_controls_2026-09-07.md) completed.
Three A_plain and three H controls exactly reproduce the earlier complete
verdict/ladder fields. A_plain CID22 held-out SROCC is **0.889095 ± 0.002369**
(sample SD across seeds); H is 0.874350 ± 0.001800. Competitive historical
training is preserved, with explicit limits on overlapping KonJND/KADID guards.

The paired H+rav1e arm adds 272 training rows and 2,010 TV pairs with exact old
prefixes. Mean CID22 delta is +0.000894 ± 0.002350. All six H fits preserve the
identity/negative-tail contract; **all six still fail all five codec floors**.
The fixed control and three rav1e candidates completed all 27 coherence cells
and received the qualifier's explicit `failed` decision. This closes the bounded
missing-supervision experiment without claiming noninferiority or deleting a
useful component.

[360 actual target loops](cleanup_target_loop_2026-09-07.md) completed with
reconstruction/bitstream parity and SSIMULACRA2, Butteraugli and fixed-B judges.
Every codec/model group fails the three-pass median-error screen. Eight passes
help, especially AVIF. The scalar-q experiment is not an encoder-RDO gain;
contended timings are excluded from performance claims. All 360 bitstreams and
pinned data/binaries match the final hash audit. The only changed pre-recorded
source is the documented wave-script failure-handling update, with unchanged
training/packing arguments.

## Verification

The repository's two documented CI configurations pass: **1,707 test executions,
zero failures, 41 explicit ignores**. These include the CubeCL CPU auxiliary
kernels. An initial blanket all-feature invocation selected CUDA and failed
because this host has no `libcuda`; the CI split below is the correct supported
runtime configuration, and CUDA/WGPU remain compile-checked by Clippy.

```bash
cargo test --workspace --no-fail-fast --all-features --exclude zensim-wasm-tests --exclude zensim-validate --exclude zensim-train-gpu
cargo test --no-fail-fast -p zensim-validate -p zensim-train-gpu --features zensim-validate/gpu-cpu,zensim-train-gpu/gpu-cpu
just clippy
just api-doc-check
```

Heavy commands ran through the existing 16-GiB/eight-job `run-heavy` wrapper.
CI-exact Clippy, API snapshot, formatting and diff checks pass. Earlier API
semver checks against `902aa68f` passed 196 checks with 58 skips and no required
semver change. No supported API removal or release was smuggled into cleanup.
The standalone target workspace's default tests/Clippy and all-feature,
all-target compilation (including JXL) pass.

Additional evidence: 604 script checks, 16 lint tests, 13 seed-argv checks
(including 11 byte-identical variants), stage reuse/interruption tests, complete
composition qualification controls, and real decoder-sidecar cache invalidation.
The spline boundary fixtures, 984-case consumed-feature census, HDR checks and
eight-arm serving matrix are recorded in their dated implementation reports.
Eight old-codec endpoint pairs re-extract all 372 features exactly; this bounded
check does not recover the extractor's unknown historical build commit.

Both board gate suites pass with 511 strict-valid source verdicts. Chromium
checks real −80…180 axes where required, unclipped points, shared scales,
selection changes, both themes, current-D identity and visible failed
qualification. The fair board is **12,160,397 bytes**, below its 12-MiB cap.
No statistic was recomputed to turn missing evidence into a pass.

## Evidence and retention

The [chronology/transcript audit](science_workflow_audit_2026-09-07.md) records
the Claude sessions, 299 memory files and 101 backups, and the 660-Markdown-file
inventory. Relevant contracts and dated research were read in depth; that
inventory is not a claim that every line of all 660 documents was independently
validated. Later split/feature/instrument rulings override the archived recipes.

The main Claude transcript is the September 7 continuation of
`9d242656-d636-45a6-9468-565163baed2d.jsonl` in the private project log directory;
the older July–August session is `8dcd6d39-57f0-4a84-97cd-6b9b08084fdb.jsonl`.
Private transcripts/configuration were not committed to the public repository.

Full runs: `/mnt/v/output/zensim/cleanup-{validation,floor-control,target-loop}-2026-09-07/`.
Private audit, source archives, logs, access instructions and hashes:
`~/tmp/zensim-science-audit-2026-09-07/`. The work/report shares and authenticated
Windows access remain configured; no credentials appear in repository records.
Pushes use the mandatory ancestry/hygiene/remote-verification owner
`scripts/safe_push.sh`; the final remote verification is retained in the private
execution log. No model/data evidence or unrelated workspace was purged.
