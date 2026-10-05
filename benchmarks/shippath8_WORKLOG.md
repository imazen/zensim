# SHIPPATH8 — protected metadata preflight and origin correction, 2026-10-05

MISSING: the human production-role decision and full composition/assessment
qualification remain pending. This fixes the two pre-landing SHIPPATH7 findings;
no protected/confirmation/T0 labels, scientific scoring, model fit or selection.
Original SHIPPATH7 inputs, tables and evidence bundle remain immutable.

P1: the Python capsule validator omitted declarations the Rust verdict owner
opens automatically. The fixed validator preflights lexical and resolved paths
for root `_MANIFEST.json`, all three supported table declaration locations,
actual corpus aliases and model `.spec.json` sidecars before opening discovered
metadata. It binds path, resolved path and SHA/presence, including absent files.
Incomplete discovery returns `labels_read:null`; it cannot assert no label read.

The existing verdict owner now has a private CLI `--print-input-paths` mode,
returning before root/table/declaration reads or scoring. It reuses the normal
input-identity inventory, so corpus slots and declaration names retain one owner.
It preflights all known candidates, parses only the allowed companion binary with
its existing loader when necessary, and inventories its bound TRAIN/source/
selection/binding paths and TRAIN declaration locations before they can be read.
The complete dynamic inventory is checked again before the wrapper creates its
output directory or invokes `--print-inputs`. The wrapper verifies every file
reported by that owner against the checked inventory, verifies stability, then
derives `labels_read` from the completed path/schema boundary checks. It carries
the checked boundary and full inventory in the emitted identity contract.

Historical scoring stages retain their input/scoring behavior; the normal
identity owner reuses the same declaration/file list and content hashes. The new
mode is private to the existing CLI, not a public library API. It performs no
new table admission, scalar scoring or selection. Existing companion numerical,
TRAIN-source and provenance checks remain in force; no gate is loosened.

Two reviewer cases were recreated with a real four-row Rust admission and real
verdict binary. The reviewed Python preflight still incorrectly accepts both
synthetic protected symlinks; the fixed validator refuses. An open tripwire
records zero sentinel opens. The actual fixed wrapper refuses before launching
its evaluator and creates no output directory. No strace is needed. Additional
regressions cover parent declarations, corpus alias declarations, model sidecars,
companion dynamic metadata, presence/byte changes, incomplete claims and
unchecked/changed owner-reported files. The Rust inventory test verifies invalid
metadata JSON is not parsed and protected root/TRAIN-alternate/model sidecars
refuse. The untampered actual complete composition control succeeds, binding 97
input candidates (70 present), with no rank/scoring output. Existing historical
primary/cross-era composition limitations remain visible.

P3: `rev5_bank.py` now derives `pairs_origin` from the actual `key_path`.
The historical default string stays identical. Eight fresh corrected bank
manifests under `/var/tmp/shippath8/verified/bank` point to the actual original
assessment keys. Corrections record prior manifest/hash, old/corrected field,
original key/hash and correction source/hash. The original extraction, producer,
assembler and decoder identities are retained; current code is not restamped as
having performed historical extraction. Original SHIPPATH7 roots were not edited.

Fresh instrument table/declaration/assessment views under
`/var/tmp/shippath8/verified/instruments` carry the updated source-manifest pins.
All eight table and key files are byte-identical. The seven unaffected confirmation
tables remain bound at their original paths. All 15 tables / 26,911 rows pass fresh
actual Rust Rev5 admission without historical replay; every consumed 420-vector
hash, ordered key hash and table/key byte hash remains unchanged, and 300 absent
positions remain NaN. `AUDIT.json` checks the correction-only delta and all pins.
`EXPOSURE_FREEZE.json` refreshes pins and remains pending, granting no label access.

Validation: 10 Python assessment tests, 4 existing full-eval stage tests, 40 Rust
verdict tests (one existing external test ignored), actual 15-table admission,
real reviewer reproductions and identity transport, plus historical default
bank extraction with identical table/key bytes and origin string. CI-exact
clippy, formatting and script lint pass. Scratch and replay commands are retained
under `/var/tmp/shippath8`; review evidence is SHIPPATH8_assets. Heavy checks used
16GiB/8-job run-heavy. No fleet, push, original-root mutation or scientific
promotion. Prior human-role and exposure decisions are unchanged.
