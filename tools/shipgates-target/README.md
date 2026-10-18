# Native scalar target validation

This unpublished tool reuses `zensim-target/src/native_probe.rs` for TRAIN-only
calibration, source-family admission, offline witnesses and the frozen 1/2/3-probe
search. It runs frozen codec-owned global encoders, then checks the emitted RGB
through a separately linked first-party decoder and an independent decode oracle.
Every probe retains its bytes, native PNG, RGB hash and work record.

The scalar-only schema explicitly leaves active/neutral-map targeting unmeasured.
The existing three-arm instrument and its public signatures are unchanged. Numeric
bars, source roles, target units and actual full-encode budgets remain unchanged.
The controller retains its nominal-direction proposal heuristic; observed ladder
inversions remain in the data. Witness admission uses actual measured scores,
never a monotonicity assumption or an interpolated feasibility claim.

Build with `just gtarget-scalar-check CONFIG` and `just gtarget-scalar-build CONFIG`.
Use an empty Cargo configuration for the committed dependency pins. An external
output directory via `CARGO_TARGET_DIR` keeps generated build output separate.
Runtime requires explicit `ZENSIM_FORMULA_REV=5` and uses the recorded encoder
and oracle binaries. The local dispatcher is `just gtarget-scalar-panel`; it
releases the shared heavy lock after each fit/evaluation phase. Missing or
incomplete phases refuse; a numerical target miss is retained and reported.

The October 18 development evidence and its limits are recorded in
[the measurement report](../../benchmarks/gtarget_2026-10-18.md). These runs
neither select a shipping candidate nor qualify spatial steering or performance.

Matched SSIMULACRA2 and butteraugli 3-norm controls use `just gtarget-peer`.
The `replay` phase replays retained JPEG/WebP candidate bytes and checks every
proposal, score, witness and selected hash; it performs zero new encodes. The
`prepare` phase measures all nine TRAIN sources at conceptual q5/q95, preserving
two independent bake runs and the frozen judge packets. Each peer tolerance is
its oriented TRAIN endpoint-median span divided by the reference candidate's
span. A TRAIN affine offset anchors q95. The `run` phase uses that fixed mapping
and the same private controller, complete calibration grids, actual witnesses
and full encode budgets. `report` invokes the existing statistical owner.

Every butteraugli packet retains its map and max/2-/3-/4-norm scalars. The fixed
CPU SSIMULACRA2 API uses the explicitly registered scalar-only artifact policy.
Reports use normalized point errors; each spec records the native-unit tolerance,
offset, orientation, fixed requests, endpoint evidence and frozen judge hash.
Peer controls never qualify active-map targeting. Active-map targeting on all
three codecs is NOT RUN: steering belongs to G-RD and first-encode map hooks
remain unvalidated.
