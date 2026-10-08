# D2 round-five admission requirements

These requirements extend the release gate map before any real KADID TERMINAL
read. This change prepares no production receipt or authorization and performs
no real read. Older receipts need regeneration and a new committed pre-read pin.

## Preparation filesystem

The required preparation/exposure layout is one dedicated, persistent,
disk-backed filesystem **F**, holding no protected corpus payload or copy.
Its device must differ from every protected device, including mounted children
under `CORPUS_ROOTS` and the original labels file's own device. The fixed roots
include `/mnt/tower/input/datasets` and
`/mnt/tower/v-datasets-archives-2026-07-22`; corpus mirrors make the tower
filesystem ineligible for F. After decoding the committed receipt, the owner
adds the labels' `os.stat` device to the protected set without opening or
hashing the label payload. That device stays protected through assessment and
result recording and is reset when the invocation ends.

Observed 2026-10-08 on dev: `/home` and `/mnt/v` report device **66304** (XFS);
`/var/tmp` reports **64512** (root ext4); `/mnt/tower` reports **103** (NFS).
Re-resolve these observations before preparing a layout; device numbers are
not constants used by the gate:

```sh
stat -c '%d %n' /home /mnt/v /var/tmp /mnt/tower
findmnt -T /home -o TARGET,SOURCE,FSTYPE
findmnt -T /var/tmp -o TARGET,SOURCE,FSTYPE
findmnt -T /mnt/tower -o TARGET,SOURCE,FSTYPE
```

A separate partition/disk or a loop-mounted ext4/XFS image can supply F,
provided its own device identity separates it from every protected store.
A bind mount of `/home`, `/mnt/v`, `/mnt/data` or `/` cannot. F must not be
tmpfs (the spent journal and ledger must survive reboot), a filesystem holding
corpus copies, or an overlay/filesystem whose files report a different device
from its mount. Persistence, absence of unregistered corpus copies and the
choice of filesystem require coordinator preparation; the opener mechanically
checks registered device separation, ancestry and regular single-link files.
It cannot certify that an arbitrary unregistered store has no protected copies.

Mount F at or beneath a trusted preparation prefix, for example
`/mnt/v/output/zensim/d2-preparation`. Paths must pass `metadata_path`: no
protected population or T0 component; no `terminal` ancestor, `_sealed` or
`holdout` component, or component starting with `labels__`. All of the following
must be regular, single-link metadata files on F:

1. The **harness checkout itself**, because `REPO` comes from the loaded source.
   It includes the registration, qualified-fit contract and profile-B bake
   `zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin`.
   Its git history must contain the committed exact receipt pin; an added jj
   workspace's shared store must be prepared consistently with that checkout.
2. The receipt and authorization; every population/prediction/prediction-receipt
   spec, panel, scorer, inspector, qualification, model, binding, gate evidence
   and E30 report. Use a **plain copy** of the pinned panel: Cargo-style hard
   links refuse under the single-link requirement.
3. The authorized checkout ledger `docs/DATA_SPLITS.md`, an explicitly selected
   journal (`--journal`) and new output. The default journal under `~/tmp`
   refuses in the current shared layout. Authorization must bind these exact
   canonical destinations; copying a ledger does not authorize its replacement.

The original labels stay on a registered protected store; never stage or copy
them onto F. No real read can proceed with the current checkout, `~/tmp`,
`/mnt/v/output/zensim` or `/var/tmp/rev4-featpot` layouts. They share protected
devices, so the admitted opener refuses before their metadata data opens.
Tower preparation also refuses through its registered corpus mirror roots.
If the receipt names labels on another device, later metadata opens and ledger
writes additionally refuse that device. Receipt parsing is necessarily earlier
than discovering this additional device; it is not a label payload read.

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
Reservation and result append use that same descriptor, with a separate
exclusive flock for each transaction. The lock is released immediately after
each transaction, including exceptional exits; no ledger lock is held during
assessment. Before each transaction and immediately before/after each durable
write, the owner compares the descriptor's device/inode and link count against
a no-follow stat of the retained canonical spelling. Unlink, rename-over,
rename-aside, changed canonical identity or extra hard links refuse as
`exposure-refused`; no success is reported for an orphaned inode. This is a
detection boundary for an uncoordinated rename during a write, not an atomic
filesystem prohibition on renames. Cooperating editors can acquire the ledger
lock between transactions, but may only append: they must preserve the canonical
inode and every prior line, including the spent reservation. Before appending
its result, the owner verifies the exact design's spent token is still present
under the transaction lock. A lock-respecting in-place rewrite that removes it
refuses as `exposure-refused`; an atomic save also causes refusal.

Retargeting the original caller alias cannot redirect either write: checks
use the retained authorized canonical name. Journal/output spellings are also
retained from admission and are not re-resolved through the caller's aliases.

The exclusive journal remains the durable one-read guard. If reservation has
created it, the design line stays spent even if a later ledger identity check
or result append refuses. A result artifact alone does not indicate successful
completion when its exposure append failed; `execute`/CLI returns
`exposure-refused`. Recover the ledger from the spent journal and preserved
result through coordinator review, never by repeating the read. After an
authorized real read, commit and land the exposure ledger from F on main.
This preparation lane creates no real authorization, migrates no real ledger,
mounts no filesystem and performs no real protected read.

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

Round-five evidence passed 56 tests: 43 prior cases and 13 new methods. No existing test
assertion or numerical threshold changed. The fixture setup adds the contract
pin and a distinct `/proc` device as a protected-store stand-in for its isolated
payload/statistic tests; it reads no real corpus. Separate regressions exercise
actual same-device refusal, transient-one-link reports and 20,000 rename flips
on disk, with kernel sentinel watches and syscall instrumentation.

The suite does not demonstrate a real prepared production layout. The round-five
independent two-filesystem helper probe accepted synthetic metadata, measured `EXDEV` for the
attempted cross-filesystem hard link and records zero sentinel opens/reads.
The same regression methods also reproduce the ledger/contract/loaded-source
defects on a source-only export of the fetched main baseline. Full commands,
hashes and resource records are retained by the round-five evidence pointer.
Its tower preparation directory proves the kernel cross-device boundary only;
it is not an eligible production preparation filesystem under the now-registered
tower corpus roots. Round-six adds unlinking atomic-save/short-lock and actual
label-device regressions; the dedicated work log owns their measured results.
