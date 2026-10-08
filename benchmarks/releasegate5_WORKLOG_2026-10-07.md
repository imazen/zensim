# RELEASEGATE5 — three P3 fixes, synthetic verification

Fresh local bookmark `quarantine/codex/releasegate5` starts from fetched
`main@origin` **`4b8f28b6924626613c28893a2ea87622d49d7737`**, which contains
the coordinator's landed `1cd8a888`. No push, real protected-label read,
production receipt/authorization, real exposure write, mount, data migration
or fleet launch occurred. [Admission requirements](releasegate5_admission_2026-10-07.md)
and [evidence pointer](releasegate5_2026-10-07.pointer.md) own the resulting contract.

## Known Bugs addressed

**Rename-time link-count P3:** the opener now refuses preparation parents or
bound leaves sharing a filesystem device with a registered protected store,
before a leaf data-open. Missing store roots reserve the nearest existing
ancestor's device; errors refuse. This enforces the review's filesystem
separation option rather than relying on another racy link-count check.
Kernel mount metadata includes protected submount devices without traversing
corpus directories. All metadata consumers, including ledger binding, supply
the protected roots.
The current shared layout refuses; an isolated canonical preparation/exposure
layout remains an operational prerequisite for a real read. It was not made here.

**Ledger re-resolution P3:** authorization captures the canonical destinations,
opens the ledger during preflight and verifies device/inode identity. Execute
retains that descriptor for reservation and result append, including its flock.
Neither phase returns to the caller's mutable ledger name. The private context
also preserves admitted journal/output spellings. The CLI adds `--journal`
so a future coordinator can explicitly select an isolated authorized journal.
Token, exclusive-create, fsync and failure-spending rules remain unchanged.

**Acceptance-input P3:** `contract_sha256` is required and checked against
the retained buffer used for contract decoding. The canonical entry loads its
private owner and dependencies by hashing and compiling the same source bytes,
bypassing pyc. Bootstrap modules validate source against executing compiled
code. Receipt source pins match the captured six-module inventory, not hashes
of disk names after import. `assessment_identity` and `v2_common` leave the
terminal import graph; their small path/hash operations reside in pinned
helpers. Third-party numerical packages and the Python environment remain
trusted environment dependencies, not silently claimed as receipt-pinned code.

Local implementation commit: `9353b8b201c2293ab6108e5c475ece82e3403e67`.
Regression/admission record: `988b6d4e83d78f6901e01fddf50ad37ab6ca3b0c`.
Protected-submount completion: `a044c61afd7906dda5e94750ac07ebcd87632077`.
The former orchestration body is the single private `_terminal_owner.py`;
`kadid_terminal_read.py` remains its canonical entry. There is no second
statistic/scoring implementation. No Rust source, dependency, serving code,
Rust public API or statistical rule changed.

## Verification

56/56 tests pass: 43 existing and 13 new methods. The earlier 12/12 focused run
checks event-recording instrumentation; the final full run also covers
protected submount-device refusal. The race and bound-payload
modules have identical ASTs to fetched main; all original test methods except
fixture setup/teardown have identical ASTs. Fixture additions supply the new
contract pin and model protected-store device separation using `/proc` as a
distinct-device stand-in. No existing assertion, numerical expectation or
threshold changed. These fixtures do not claim a physical production layout.

New probes cover same-device and nested-mount refusal before receipt access, a deterministic
transient-one-link report,20,000 S6 rename iterations, ledger alias/regular
replacement after preflight and alias replacement before result append,
contract mutation/missing pin, post-import disk re-pinning, retained-source
compilation, stale bootstrap code and real child import-side-effect tripwires.
The final full run measured 459,754 flips with zero instrumented harness data-opens
and zero inotify sentinel opens/reads. The final focused run's own counts are
retained separately. Kernel event coalescing can undercount nonzero stress
events; the independent syscall oracle also records zero.

A separate actual-filesystem proof uses two fresh synthetic directories on
devices103 and66304. Approved metadata opens correctly; attempted cross-device
hard-link creation returns `EXDEV` (18). Sentinel opens/reads are zero.
No real corpus participates in this probe.

Seven discriminating regression methods run on the source-only main export
fail 7/7. Ledger aliases/replacements after preflight record two unauthorized
opens and one read; redirecting the result append records one open. Changed
contract, missing contract pin, post-import disk pin and transient-one-link
cases each record one protected synthetic label/sentinel open and read.
All corresponding fixed events are zero. Baseline raw logs are retained;
expected negative-control failure is not a failed fixed-code gate.

CI-exact Clippy, full fmt check, scoped Ruff F and diff check pass.
Script lint checks 863 scripts, all runnable. Its tracked-file conflict/hygiene
scan uses a scratch index restricted to source/docs/config, excluding 1,312
payload/other paths without reading them. The normal index is untouched.
Panel Rust source is unchanged; canonical panel SHA remains
`a669bec560f7d376ed14e022bec95f8259a62be270809e7d830bb4d5e80967b0`.
Full positive panels and original independent-statistic tests pass; legacy
36-case parity and17 Rust panel tests were not separately rerun this round.

All heavy commands were serialized under run-heavy. Full suite:
`rc=0 107s | peak-RSS 0.41GiB | min-avail 46223MiB | peak-load 2.61`.
Final focused module:
`rc=0 31s | peak-RSS 0.36GiB | min-avail 46780MiB | peak-load 2.24`.
Negative controls:
`rc=1 33s | peak-RSS 0.37GiB | min-avail 47055MiB | peak-load 2.25`.
Clippy:
`rc=0 54s | peak-RSS 0.93GiB | min-avail 45344MiB | peak-load 6.41`.
These are shared-machine verification records, not performance qualification.

No real D2 verdict, final-model qualification or release approval is implied.
