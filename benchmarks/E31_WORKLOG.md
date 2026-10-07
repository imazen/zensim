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
