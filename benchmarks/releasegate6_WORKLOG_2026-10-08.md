# RELEASEGATE6: ledger transactions and preparation layout

Base is the exact reviewed `quarantine/codex/releasegate5` tip
`42972b5a3281342fac6a747f31beacf6a7a15781`. This lane fixes the two named
findings only. No real KADID TERMINAL, sealed, T0, AIC or human-label payload
was opened; no real receipt, authorization, journal or ledger was prepared or
changed. The native panel operates on synthetic values. No Rust, scientific
contract, threshold, population, model or default metric changed. No push.

## P2: detect a replaced canonical ledger and use short locks

The authorized descriptor and canonical spelling remain bound at preflight.
Each reservation/result transaction acquires an exclusive flock, compares
`fstat` with a no-follow stat of the canonical name, and releases its lock in
`finally`. The checks require the same device/inode and one retained/canonical
link; they also reject the labels' protected device. A second check immediately
before and after the fsynced append catches unlinking rename-over, not merely
renaming the old ledger aside. A replaced name or orphan refuses as
`exposure-refused`; result append failure can no longer return successful
`execute`.

The lock is absent during assessment, although the admitted descriptor remains
open. A cooperating writer can lock between transactions; a writer that
atomically replaces the ledger instead of preserving its inode causes refusal.
The exclusive spent journal, fsync ordering and no-repeat rule remain unchanged.
If the journal was created, later refusal leaves the design spent. A persisted
result alone is not completion when the ledger append failed; the caller gets
`exposure-refused`. This detects an uncoordinated rename during a write; it does
not make filesystem rename and append one atomic operation.

Regressions exercise actual unlinking `os.replace` before reservation, after
reservation, after journal creation and during durable write. An independent
descriptor successfully acquires nonblocking flock during real synthetic
assessment and after result append. Another regression raises while the
retained descriptor stays open and confirms explicit unlock on exception.
The previous rename-aside test now requires refusal, a stronger expectation
directly requested by this brief. Alias-retarget tests still require exact
authorized-inode writes and zero unauthorized inode events.

## P3: precise separate preparation/exposure requirements

The [requirements](releasegate5_admission_2026-10-07.md) now specify a dedicated
persistent disk-backed filesystem F, separate from every protected device.
They enumerate the checkout/git history, receipt/authorization, all pinned
specs/tools/models, profile-B bake, canonical ledger, explicit journal and
new output on F. Panel copies must have one link. Labels stay on a registered
protected store; never copy them onto F. They distinguish mechanical refusal
from operational persistence and absence-of-unregistered-copy requirements.

Both named tower corpus mirror roots are registered. The labels' own device
is added by `os.stat` after decoding the committed receipt, without opening
or hashing labels; the additional set persists through the invocation and
does not leak afterward. The real same-device labels regression refuses
preflight with zero label inode open/read events. The synthetic payload/statistic
fixtures explicitly model a distinct device with `/proc`, as earlier isolation
fixtures did; real device-boundary regressions remove that stand-in.

Metadata-only observations on 2026-10-08: `/home` and `/mnt/v` device66304;
root-backed `/var/tmp` device64512; tower NFS device103. Resolving commands
and prohibitions on bind mounts, tmpfs, corpus-copy filesystems and incompatible
overlay layouts are in the requirements. No mounts, real ledger migration or
real exposure occurred. The current shared layout intentionally remains refused.

## Measured validation

`just releasegate-tests`: **64/64 pass**, including all56 preceding tests
and eight new methods. Exact command:

```sh
flock ~/tmp/zensim-paper/rev4/heavy.lock \
  ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- \
  env TMPDIR=$HOME/tmp/releasegate6 OPENBLAS_NUM_THREADS=1 \
  ZEN_PANEL_BIN=/mnt/v/output/zensim/releasegate2-2026-10-07/bin/panel \
  just releasegate-tests
```

Receipt: `rc=0 93s | peak-RSS 0.41GiB | min-avail 45737MiB | peak-load 5.28`.
This is suite process memory, not production scorer RSS qualification.
The initial63/64 run remains preserved: the scope unit accidentally used the
fixture's simulated label device; restoring the real registration method
fixed setup without changing its assertion or any numerical expectation.

Three identical ledger behavior probes run against a source-only export of
the exact base fail as expected: both rename-over refusals are absent, and
the assessment-time writer blocks. The control runner requires exactly two
test failures and one error, with no skips, then returns success for reproducing
the known defects. Receipt:
`rc=0 8s | peak-RSS 0.36GiB | min-avail 46065MiB | peak-load 3.86`.
The export contains only the named terminal source/fixture/contract files;
no dataset tree or protected payload is exported.

Scoped Ruff F, Python compilation and reverse git-apply whitespace/patch
validation pass. No assertion, statistical threshold or numerical expectation
was relaxed. No Rust build or Cargo target was created; Rust checks/remote CI
are not claimed. Reproducible commands are in
[the task recipes](releasegate6.just). Raw logs, source snapshots, exact panel
SHA256 and loaded-source pins live under
`/mnt/v/output/zensim/releasegate6-2026-10-08/`; the completion marker records
the final verification and archive hashes. All heavy commands serialize
through the coordinator's shared heavy.lock plus run-heavy.
