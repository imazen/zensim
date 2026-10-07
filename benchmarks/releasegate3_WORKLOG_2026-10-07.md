# RELEASEGATE3 bound metadata I/O — 2026-10-07

Local fix on `quarantine/codex/releasegate2`, extending reviewed tip
`e0e40a14136dcea52b0baa51544c883bc08275a0`. No push, real protected-label
read, production authorization, exposure-ledger change, fleet action or serving
switch. Synthetic fixture labels only. Final source, logs and commit identity
are retained at `/mnt/v/output/zensim/releasegate3-2026-10-07/VERIFY.json`.

## Known Bugs

**Remaining review P1: symlink retarget between admission and open — fixed.**
The old metadata helper checked lexical/resolved ancestry but returned a mutable
pathname for `Path.read_bytes`. A late alias retarget opened synthetic terminal
labels before authorization. The new reviewer regression fails on the exact
reviewed tip: one synthetic sentinel open, zero spent records. It passes on the
fix with zero sentinel opens and zero spent records. The reviewer source is
`/var/tmp/releasegate2-review/alias_race.py`; the committed regression preserves
its real `io.open`-boundary hook and stable-alias control, demanding zero opens.

## Bound-open contract

Metadata admission returns exactly the resolved spelling whose ancestry passed
the existing lexical/resolved preparation policy. Stable prepared aliases remain
supported. Every subsequent metadata read/hash opens from `/` through directory
handles: `O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC` for every ancestor, then
`O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK` for the leaf. The leaf must be a regular
file before any read. A symlink replacing a not-yet-open ancestor or leaf
refuses. A directory handle already acquired survives pathname renaming without
redirecting the subsequent read. Retargeting an original alias cannot redirect
the resolved handle walk. No pathname-open fallback exists.

This is a Linux orchestration contract, using directory-fd opens and
`/proc/self/fd`; unsupported platforms refuse rather than weakening admission.
This fix addresses pathname/symlink retargeting. It does not establish
immutability against arbitrary in-place writes to a regular file or privileged
mount changes. Content pins still verify each consumed metadata buffer.

Receipt and authorization use bound reads. Receipt parsing still consumes the
verified committed bytes. Registration/code/model/gate/report/binding hashes and contract reads
use bound handles. Pinned JSON and TSV tables hash and parse the same retained
buffer; no second mutable-name open lies between checksum and decoding. The
existing raw `sha` convenience helper is used only by synthetic fixture setup;
no production metadata consumer calls it.

The canonical external checkpoint inspector and selected model each get a
hash-checked, retained handle. The child executes/reads those inherited fds
through `/proc/self/fd` with explicit `pass_fds`, covering its metadata decode
without reopening the admitted inspector/model names. The real inspector owner
(`zensim-validate/examples/inspect_qualified_checkpoint.rs`) reads the explicit
argument as bytes; it does not depend on a filename extension. A synthetic
inspector actually reads the inherited model and asserts its exact original
bytes after both mutable names are retargeted.

Ledger reservation/result append, exclusive journal creation, result creation
and result hashing also use the bound opener. Journal/result creation is
exclusive: an existing leaf or inserted symlink cannot be overwritten. Ledger
flock and before-label fsync/spent semantics remain unchanged. Actual terminal
labels remain the sole authorized exception, after durable reservation; no real
label path was opened while preparing or testing this change.

## Verification

```bash
TMPDIR="$HOME/tmp/releasegate3" \
ZEN_PANEL_BIN=/mnt/v/output/zensim/releasegate2-2026-10-07/bin/panel \
../scripts/run-heavy --mem 16G --jobs 8 -- just releasegate-tests
```

30/30 synthetic tests pass, including every prior23 test. The complete prior
23-test module has an identical AST; no assertion/threshold was relaxed.
New probes cover the reviewer's late alias, admitted-alias retarget, leaf
retarget across receipt/authorization/registration/JSON/TSV/model/inspector,
mutable ancestors before and after directory acquisition, retained-buffer
hash+parse, reservation ledger/journal races and the actual inspector child
reading the inherited model. Protected-sentinel tripwires inspect successful
`os.open` handles by device/inode as well as the original `io.open` boundary.
Refusals leave no journal/output/ledger marker. Stable alias preflight passes
without labels or spending. Full positive assessment and receipt-replacement
controls remain green.

Final harness wrapper:
`rc=0 32s | peak-RSS 0.42GiB | min-avail 41769MiB | peak-load 11.43`.
CI-exact clippy passes with denied warnings:
`rc=0 60s | peak-RSS 0.92GiB | min-avail 43055MiB | peak-load 18.31`.
Full fmt check and scoped Ruff F pass. Script lint checks852 runnable scripts.
All heavy commands were serialized under16GiB/8-job caps; these are software
checks, not quiet-box runtime/memory qualification.

The hash-pinned canonical panel is reused from round two, SHA-256
`a669bec560f7d376ed14e022bec95f8259a62be270809e7d830bb4d5e80967b0`.
No Rust/panel source or dependencies changed. The existing36-case legacy panel
parity is rerun at1e-9, conditional on canonical emitted logistic predictions.
The round-two17 Rust panel tests remain the prior verification, not a new
round-three execution. Final production fits/gates, companion predictions,
complete original stimulus mapping and owner authorization remain prerequisites.
No real D2 verdict or production qualification was performed.
