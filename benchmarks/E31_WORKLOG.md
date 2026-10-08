# E31 implementation gate — 2026-10-07

**BLOCKED before arm packaging.** The implementation brief requires control
parity first and says to stop if it fails. E30's 40 completed nA3 cells are
frozen; executable E31/control parity is not established. No replacement
control was selected, no E31 fit ran, and nothing was pushed or enqueued.

Registration: [E31](e31_upiq380_registration_2026-10-07.md), SHA-256
`016d168d04caa9f2ae1853fed25602f845e081fb009546a6ece6ce4d8ae563a7`.
Lane base: zensim `d3b91625`. Local implementation:
`6b525cf2086f1790c9a0125a34a9d74c484f9623`.

## Control freeze

The existing E30 owner now exposes `completed-control-pins`. It addresses
exactly four registered sources × seeds 0–9 without enumerating historical
cells. This freezes the nA3 arm, not E30's E24 comparison control.

All 40 cells passed result/fleet-byte bindings, ordered 420-feature identity,
D1 role and seven-table admission receipts, 120 × 50,000 budget and final-119
selection checks. The pinned canonical model inspector verified every selected
bake's embedded final-119 provenance, inputs, uniform sampler, fold-rotated
sample seeds, initialization seeds, logical width 1853 and minibatch 1.
The full embedded argv and receipts are retained for the remaining parity
audit; this does not certify an arm which cannot yet execute.

Control program archive:
`6d491eb165a00b70297de0864ec2e749818ca81491207a7da3fa6e36c2028cd0`.
Control data archive:
`8f2dae5e024670c02bbb81e5ed18254a5132f4423fc672f6aeb1059560bba084`.
All packed program inventory entries and both archive hashes were verified.
All 320 installed control files match the existing Tower checksum index;
three deterministic random Tower files were independently hashed.

## Measured admission blocker

E30 trainer SHA-256:
`9954507d5ab39e01ee961ac495856098d71eeef7b95b686f18a586a50e26524b`.
The actual pinned binary refuses the round-two registered UPIQ fit manifest
with exit 2: `malformed feature_set_id`. Its declaration has a null family
identity and an explicit 420-ID native HDR subset. `strace` records zero
opens of the UPIQ or SafeSyn training payloads in this check. The original
version named by the brief also declares a null `feature_set_id`.

A separate fully synthetic two-table check, with valid registered IDs and
distinct decoder declarations, exits 2 with `mixed decoder declarations`.
It opens only synthetic table payloads. The strict Rust owner rejects distinct
decoders even when the IDs and Rev5 agree. The existing Python strict owner
also requires the SDR family ID, decoder era, D1/oracle role and row-selection
contract; the existing HDR helper admits only E26's teacher population.

Thus the current strict executable cannot consume the registered native HDR
transport. Neither assigning the RGB8 decoder contract to EXR inputs nor
historical replay is an admissible parity fix. A truthful native-HDR training
declaration and compatible strict mixed-ingress admission must be resolved,
then baseline behavior/program identity must be audited against this freeze.
If that resolution changes the required fit binary/program identities, the
registration leaves any fresh matched 40-cell control decision to the
coordinator. This lane did not make that decision.

The UPIQ HDR-only derivative still has null producer commit and
`qualified_provenance=false`; its separate owner disposition remains required
before fitting. D3 authorizes the population, but does not fill that missing
provenance. No human table, original mixed CSV, image or protected payload
was opened by this lane's admission checks.

## Verification and unimplemented scope

Four grid tripwire tests pass: AIC fold, duplicate cell, missing cell and
unregistered seed all refuse before payload hashing, model inspection or cell
enumeration. Both actual-binary refusal checks pass. Ruff checks the changed
Python owners; the two new Python files pass format check. `just lint-scripts`
passes. The 27 existing SHIPPATH regression tests also pass with the pinned
E30 trainer selected through `SHIPPATH_TRAINER`; their bounded training uses
synthetic rows only. The first run without that override failed two tests
because the fresh workspace has no debug trainer. No expectations were changed.
No Rust source changed and no crate build/test or clippy run is claimed.

Per the first-gate stop rule, the arm route, zenmetrics profile/workspace,
program/data packs, declaration, image and real-executor smoke, harvest
adapter, paired scorer, launcher/authorization and scorer root are **not
implemented**. No arm manifest or launch authorization exists. The zensim
`e31` workspace remains for coordinator disposition; no workspace was removed.

Reproduction commands (run under the normal heavy-command wrapper, with
`TMPDIR` on disk; freeze and refusal outputs require fresh paths):

```sh
just e31-control-tests
just e31-control-freeze /mnt/v/output/zensim/shippath11-2026-10-07 /var/tmp/rev4-featpot/e30-results/cells <fresh-freeze.json>
just e31-pinned-admission /mnt/v/output/zensim/shippath11-2026-10-07 /mnt/v/output/zensim/upiq380-rev5-r2-2026-10-07 <fresh-evidence-directory>
```

The receipt paths and hashes are in
[the evidence pointer](e31_preparation_2026-10-07.pointer.md).

## Extension after the coordinator's control decision

The preceding gate records the first implementation stop. The coordinator
subsequently selected a shared fresh v40 control for E29/E31/E32 before
any arm fit. This lane now implements the E31 trainer extension on
`main@origin = 13ece3619b8f74656aef2066211e74bd69cdefa1` in the retained
`e31` workspace. The combined fleet package is a later coordinator step.
During validation, origin advanced to
`ced5090f74447e376bae2d638739b33836a45fe7`. The local chain was rebased
onto that tip; an append conflict in the justfile was resolved retaining
all SPEEDQ and E31 recipes. Every trainer/recipe source hash and Cargo.lock
remained unchanged, and the rebuilt trainer binary was byte-identical.

The extension commits after rebasing are `cbedc5a7` (native admission),
`2a83b46c` (original row identities), `da4f7b17` (ingress restrictions,
transport tests and complete control comparison) and `dc0fd609`
(research receipt provenance).

The Python recipe owner accepts exactly the registered nA3 recipe plus
`:uh4`, head N, and an explicitly supplied UPIQ fit table and label-gap
disposition. The fit-only owner checks the round-two immutable manifest,
330 ordered fit conditions, 26 original reference-byte identities, their
modulo-five split and pair-key bindings, and the exact by_v2fy 420 IDs.
The membership-derived train weight is `4.34410740924913`; validation
weight is zero and pairing is pooled rank-only. The development population
is never added to the training or checkpoint-selection groups.

The trainer has a private, opt-in UPIQ admission module. Without
`--upiq-label-disposition`, it calls the original table admission owner.
With the flag, it checks every table declaration before payload access,
admits inherited SDR groups through the same owner, and binds the native
leg to its exact manifest/table/key hashes. Its Float64 loader retains
the original bits; the extension pads only absent columns f1825–f1852
with NaN to match logical width 1853. The existing keep mask and scaler
then handle the projection. Targets are not clipped. No optimizer,
scaler, sampler, RNG, loss arithmetic or public Rust API changed.

Native EXR input keeps its own declared input contract and producer
pins in `zentrain.repro.table_admission.upiq380`; it does not receive
the SDR producer identity. The unresolved legacy label producer remains
null and combined provenance remains false. Research result receipts
use `e31-native-hdr-research-training-cell-v1`.

Actual E31 fitting still requires the owner's explicit disposition of
the label gap. D3 alone is insufficient. The decision format requires
`schema=e31-upiq-label-disposition-v1`,
`decision_id=E31-legacy-HDR-label-producer-gap`, `state=approved`, a
nonempty `decided_by`,
`allowed_use=registered-E31-research-training`, the pinned fit-manifest
and legacy-label hashes, and `accept_unresolved_producer=true`.
Test decisions are synthetic fixtures only; this lane created no real
approval and fitted no UPIQ arm.

Reproducible owners added to the justfile:

```sh
just e31-training-tests
just e31-fit-key-check <pinned-upiq-fit.parquet>
CARGO_TARGET_DIR=<own-target> just e31-crate-tests
CARGO_TARGET_DIR=<own-target> just e31-build-trainer
just e31-extended-admission <trainer> <pinned-upiq-fit.parquet> <fresh-evidence>
just e31-control-cell <D1-root> <binary-dir> <fresh-cell> <disk-scratch>
just e31-control-parity <E30-kadid-s0> <fresh-cell> <binary-dir> <inspector> <fresh-parity>
```

The full control command runs under `run-heavy --mem 16G --jobs 1`,
one Rayon/OpenBLAS/OMP thread and v3, 120 epochs × 50,000 pairs,
initialization seed 1101, sample seed 101 and fixed final epoch 119.
The parity owner retains original bakes and compares every byte after
the existing strip owner removes only `zentrain.repro`. It separately
checks all nonvolatile reproduction fields, content-bound inputs,
feature bytes and complete result receipts. Only timestamps, runtime
locations and source/build location fields are normalized.

### Measured local verification

The first full-budget kadid seed 0 control and the rebuilt final Rust
extension's full-budget control both passed all four comparison gates:
complete non-repro bake bytes, nonvolatile repro, keep-list bytes and
complete result receipts. Each stripped bake is 215,978 bytes with SHA-256
`4fc21dc98d83cdcfba25f38623921e90595e976c75ad2fa8a4f8b2eb7f1a95f2`.
Original bakes retain their actual source/path/timestamp provenance, so
their full-file hashes differ; raw-bake byte identity is not claimed.
The final Rust run's resource record is:

```text
run-heavy: done rc=0 290s | peak-RSS 0.96GiB | min-avail 43575MiB | peak-load 19.12
```

Six actual-binary negative cases pass with exit 2 and zero feature
payload opens recorded by strace: missing extension disposition,
same-cardinality different feature subset, development manifest,
AIC source, within-reference pairing and an absolute-regression loss.
The default native-table refusal remains `malformed feature_set_id`.
The pinned fit-key-only check confirms all membership/split bindings and
the exact registered weight without opening a feature/development payload.

Nine E31 Python tests and 27 SHIPPATH regression tests pass. The targeted
crate run passes 285 library tests and 29 trainer tests; one pre-existing
library test remains ignored. No test expectation or ignore attribute
was changed. Four native trainer tests cover f64 Parquet fallback,
unclipped targets, bit-preserving padding, rank-only recipe enforcement
and the distinction between D3 and label-gap disposition. CI-exact
workspace clippy, scoped workspace fmt check, Ruff and the 855-script
syntax audit pass. No dependencies were added.

The initial key check exposed the manifest's `references` field and the
fit split's retained original row IDs; those implementation errors were
corrected. An early negative run accidentally selected the previous binary
while the final build was still running; its failed evidence is retained.
The completed final binary was then checked separately and passed.

The build and evidence pins are in
[the extension evidence pointer](e31_extension_2026-10-07.pointer.md).
The measured control covers one of the 40 fold/seed positions, with
repeated full fits for source revisions. It does not replace the
coordinator's registered fresh 40-cell v40 control. No fleet package,
image, declaration, push, enqueue, launch or UPIQ arm fit was performed.

The additional full-budget control from the rebased workspace also passed
all four gates, with the same stripped SHA-256 above. Its original selected
bake SHA-256 is
`1fded1a12d76ca2201581459c98aeee48524f22d76105bb07ae23c3395c21d88`.
Post-rebase scoped fmt and the 857-script audit pass. Resource record:

```text
run-heavy: done rc=0 282s | peak-RSS 1.04GiB | min-avail 44673MiB | peak-load 14.63
```
