# D2 round-five admission requirements

These requirements extend the release gate map before any real KADID TERMINAL
read. This change prepares no production receipt or authorization and performs
no real read. Older receipts need regeneration and a new committed pre-read pin.

## Preparation filesystem

Every metadata payload, executable, model, ledger, journal and result must be
on a filesystem whose device identity differs from every registered protected
store in `CORPUS_ROOTS`. The opener checks the retained parent directory and
the retained leaf identity before the data open. Kernel mount metadata also
reserves the devices of mounted children beneath each protected root; no
corpus directory traversal is used. Missing store paths reserve
the filesystem of their nearest existing ancestor; inspection errors refuse.
The single-link rule remains an additional check, not the hard-link boundary.

This rejects the reviewer's transient `st_nlink == 1` race even when every
link-count observation says one. A hard link from a registered protected store
into an admitted preparation filesystem is impossible: the kernel returns
`EXDEV`. A bind mount of the same filesystem does not meet the device test.
Path spelling, mountpoint names and post-read rejection are insufficient.

The current shared local layout is intentionally refused. Before a real read,
the coordinator must prepare a genuinely separate filesystem under an existing
trusted preparation prefix, such as `/mnt/v/output/zensim/d2-preparation`, and
freeze the canonical ledger/journal destinations there. Copying a ledger does
not establish authorization to a new canonical exposure destination. Keep all
protected stores registered; changing store locations requires provenance and
policy updates. No mount, real ledger migration or corpus copy occurred here.

Read-only source bootstrapping is part of the trusted Python program launch;
it does not discover or open dataset labels. The filesystem rule applies to
the terminal owner's metadata I/O, including the authorized ledger handle.

## Receipt and loaded source identity

Keep the existing receipt schema and scientific fields. Add
`contract_sha256`, the SHA-256 of the exact qualified-fit contract JSON.
Contract decoding consumes the same buffer that passed this pin.
`code_sha256` must equal the complete six-entry loaded-source inventory:

- `kadid_terminal_read.py`
- `_terminal_acceptance.py`
- `_terminal_owner.py`
- `_terminal_bound_io.py`
- `v2c_labels.py`
- `zen_stats.py`

To print the inventory without opening data, contract or labels:

```sh
python3 -c 'import sys,json; sys.path.insert(0,"scripts/rev4_featpot"); import kadid_terminal_read as o; print(json.dumps(o.CODE_SHA256,sort_keys=True,indent=2))'
```

The canonical entry loads the private owner, label adapter, I/O helper and
statistics shim by hashing and compiling the same retained source bytes;
cached bytecode is bypassed. Bootstrap modules verify source/compiled-code
agreement against their executing module code. Preflight compares the receipt
to captured source identities, rather than hashing mutable disk names after
import. The terminal path imports neither `assessment_identity` nor
`v2_common`: its local path guard is in the pinned I/O helper and its label
hash uses `hashlib`. The Python interpreter, standard library and installed
numerical dependencies remain the trusted execution environment.

## Authorized destinations

Authorization resolves the ledger and journal once and compares their canonical
destinations with the caller's. During `execute`, it binds the authorized ledger
descriptor and verifies device/inode identity before returning from preflight.
Reservation and result append use that same descriptor; it remains open and
locked through assessment. Retargeting the original alias or replacing its
regular file cannot redirect either write. Journal/output spellings are also
retained from admission and are not re-resolved through the caller's aliases.

The canonical CLI adds an explicit journal override:

```sh
python3 scripts/rev4_featpot/kadid_terminal_read.py \
  --receipt "$COMMITTED_PIN" --authorization "$AUTH" \
  --ledger "$AUTHORIZED_LEDGER" --journal "$AUTHORIZED_JOURNAL" \
  --output "$NEW_RESULT"
```

The existing default journal remains for compatibility and will refuse if its
filesystem violates isolation. The one-read token, exclusive journal creation,
fsync order, population, statistical thresholds and post-error spending are
unchanged. Unsupported platforms, device conflicts, pin failures and changed
destinations refuse; there is no weaker fallback.

## Synthetic verification

The suite passes 56 tests: 43 prior cases and 13 new methods. No existing test
assertion or numerical threshold changed. The fixture setup adds the contract
pin and a distinct `/proc` device as a protected-store stand-in for its isolated
payload/statistic tests; it reads no real corpus. Separate regressions exercise
actual same-device refusal, transient-one-link reports and 20,000 rename flips
on disk, with kernel sentinel watches and syscall instrumentation.

The suite does not demonstrate a real prepared production layout. An independent
two-filesystem probe accepts synthetic metadata, measures `EXDEV` for the
attempted cross-filesystem hard link and records zero sentinel opens/reads.
The same regression methods also reproduce the ledger/contract/loaded-source
defects on a source-only export of the fetched main baseline. Full commands,
hashes and resource records are retained by the round-five evidence pointer.
