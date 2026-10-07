# RELEASEGATE4 bound labels, panel and inode admission — 2026-10-07

Local implementation on `quarantine/codex/releasegate2`, extending reviewed
tip `8e81659b2f9e3673edd81e29047cffc8dae8d9cd`. No push, real protected
payload read, production authorization, real exposure-ledger change, fleet
action or serving switch. All label-bearing probes generated synthetic data.
Evidence is indexed by [the pointer](releasegate4_2026-10-07.pointer.md).

## Known Bugs

The round-three review's label-reopen P1, panel-path/fallback P1 and hard-link
P2 have regression coverage and are fixed for the tested substitution cases.
The separate P3 findings about ledger destination re-resolution and complete
loaded-code/contract identity remain outside this change. This record does not
qualify the harness for a real D2 read or establish any production-model result.

## Bound payloads

`v2c_labels.load_label_rows` reads the label file once, hashes that retained
buffer and parses that same buffer. CSV/TSV use `BytesIO`; JSON uses the bytes
directly. The optional pinned pairs table follows the same contract. Renaming
the file or retargeting its alias after hashing cannot change decoded values.
The authorized label read still follows durable reservation; metadata admission
does not read label payloads.

The terminal owner opens and re-verifies the panel before reservation, retains
its descriptor through assessment and executes `/proc/self/fd/N` with explicit
descriptor inheritance. The direct signed-quality call and every statistics
shim call share that descriptor. No terminal call returns to the panel's
mutable name. A missing explicitly selected `ZEN_PANEL_BIN` now refuses; the
generic shim retains release/debug discovery only when no override or bound
descriptor was supplied.

Metadata admission compares `(st_dev, st_ino)` against the receipt's label
identity before reading bindings, evidence or outputs. Before any metadata
leaf data-open, the private Linux helper requires a regular file with exactly
one link, binds an `O_PATH` descriptor and checks its identity against the
preceding stat. The data handle opens through that kernel-held descriptor,
not the caller's leaf name. A hard link inserted before identity acquisition
refuses without a sentinel data-open. Insertion afterwards cannot redirect the
held inode. Exclusive outputs retain `O_EXCL` and never open an existing leaf.

The helper joins the receipt's source inventory: new receipts require four
source pins, including `_terminal_bound_io.py`. Existing three-file receipts
must be regenerated and committed before any future coordinator authorization.
No real receipt or authorization was prepared in this lane.

This is a Linux orchestration contract. `O_PATH` is an identity handle and
produces no kernel `IN_OPEN`/`IN_ACCESS` payload event. The descriptor strategy
binds the object across name replacement; it does not freeze arbitrary in-place
writes to that inode or privileged mount changes. Original-file content pins
remain required. Scientific thresholds, panel Rust code, serving code, Rust
public API and dependencies are unchanged.

## Regression evidence

The complete suite passes 43/43 tests: the original 23 tests, the existing
seven metadata-race tests and 13 new regression methods. Both original test
modules have identical ASTs to the reviewed tip. Expectations were not relaxed.
The new tests use Linux inotify watches on sentinel inodes, independent of
Python open hooks and alias spellings, and require zero opens and reads.

New coverage includes CSV/TSV/JSON label alias and rename replacement after
hashing, pinned pairs replacement, full authorized synthetic assessments whose
verdict would change with substituted labels, panel alias/rename/removal after
reservation, missing explicit panel with a present fallback, changed panel
before reservation, reviewer H1/H2 hard links and insertion at both leaf-binding
boundaries. Three full panel substitution cases independently verify the
descriptor hash and all 113 statistic subprocess commands/descriptor inheritances.

The exact new test module was also run against a source-only export of the
reviewed tip. All 13 methods fail there (18 subcase failures and one error).
Label/pairs/H1/H2 sentinels record one open and one read per corresponding probe;
panel alias and rename probes record 226 opens/reads from 113 substituted shell
executions; removal reaches a fallback with two opens/reads. The later identity
hook test also fails because the old implementation lacks that binding boundary;
its failure alone is not evidence of a sentinel open.

The canonical panel is reused from round two, SHA-256
`a669bec560f7d376ed14e022bec95f8259a62be270809e7d830bb4d5e80967b0`.
The 36-case legacy panel parity check passes at 1e-9, conditional on Rust's
emitted logistic-rescaled predictions. It does not independently validate
the logistic fit. No real terminal assessment or production qualification ran.

Reproduction uses the committed `releasegate-tests`,
`releasegate-bound-payload-tests` and `releasegate-panel-parity` just recipes.
Set `ZEN_PANEL_BIN` to the pinned canonical binary and run heavy commands
serially through `../scripts/run-heavy`. Full commands, raw logs, source hashes
and resource records are in the evidence index.

Final checks pass: CI-exact Clippy with denied warnings, full formatting check,
scoped Ruff F and script lint (854 scripts). Tracked-file lint uses the
source/docs/config-only scratch index described in the evidence pointer,
keeping payloads unopened. The shared machine's resource records are retained;
no timing or memory improvement is claimed.
